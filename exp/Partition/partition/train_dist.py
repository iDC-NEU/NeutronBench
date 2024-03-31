import argparse
import socket
import time
from contextlib import contextmanager
from dgl.nn.pytorch import GraphConv
import dgl
import dgl.nn.pytorch as dglnn

import torch.distributed as dist
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import sklearn.metrics


def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys feat and label of a set of nodes onto GPU.
    """
    # batch_inputs = (
    #     g.ndata["feat"][input_nodes].to(device) if load_feat else None
    # )
    gather_start = time.time()
    tmp_input = g.ndata["feat"][input_nodes]
    tmp_labels = g.ndata["label"][seeds]
    gather_time = time.time()-gather_start
    
    trans_start = time.time()
    batch_inputs = tmp_input.to(device)
    batch_labels = tmp_labels.to(device)
    trans_time = time.time()-trans_start
    return batch_inputs, batch_labels , gather_time, trans_time



class Model(nn.Module):
    
 
    def __init__(self,num_i,num_h,num_o):
        super(Model,self).__init__()
        
        self.linear1=nn.Linear(num_i,num_h)
        self.relu=nn.ReLU()
        # self.linear2=torch.nn.Linear(num_h,num_h) #2个隐层
        # self.relu2=torch.nn.ReLU()
        self.linear3=nn.Linear(num_h,num_o)
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu2(x)
        x = self.linear3(x)
        return x


class GCN(nn.Module):
    def __init__(self,
                    in_feats,
                    n_hidden,
                    n_classes,
                    n_layers,
                    activation,
                    dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_hidden = n_hidden
        self.activation = activation
        self.n_classes = n_classes
        # input layer
        # self.layers.append(nn.li)
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    # def forward(self, blocks, x):
    #     h = x
    #     for i, (layer, block) in enumerate(zip(self.layers, blocks)):
    #         h = layer(block, h)
    #         if i != len(self.layers) - 1:
    #             h = self.activation(h)
    #             h = self.dropout(h)
            
    #     return h

    def forward(self, mfgs, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i > 0:
                # h = self.activation(h)
                h = self.dropout(h)
            # print(h[:mfgs[i].num_dst_nodes()].shape)
            h_dst = h[:mfgs[i].num_dst_nodes()]
            # h = self.layers[i](mfgs[i], (h, h_dst))
            h = layer(mfgs[i], (h, h_dst))
        return h

    def inference_test(self, g, x, batch_size, device,nodes_tmp,model):
        
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )

        y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
        y = th.ones((g.num_nodes(), self.n_classes),dtype=th.float32)
        print(f"|V|={nodes.shape[0]}, eval batch size: {batch_size}")
            # print('y:',y.shape)
        sampler = dgl.dataloading.NeighborSampler([10,10])
        # sampler = dgl.dataloading.NeighborSampler([4,4])
        # sampler = dgl.dataloading.NeighborSampler([-1,-1])
            # sampler = dgl.dataloading.NeighborSampler([10,25])
        dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes_tmp,
                sampler,
                batch_size=len(nodes_tmp),
                shuffle=False,
                drop_last=False,
            )
        


        for input_nodes, output_nodes, blocks in (dataloader):
                h = x[input_nodes].to(device)
                for i, layer in enumerate(self.layers):
                    block = blocks[i].to(device)
                    h_dst = h[: block.number_of_dst_nodes()]
                    print(block.number_of_dst_nodes(),' ',block.num_dst_nodes())
                    h = layer(block, (h, h_dst))
                    if i != len(self.layers) - 1:
                        # h = self.activation(h)
                        h = self.dropout(h)

                y[output_nodes] = h.cpu()
        g.barrier()
        return y

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.

        Distributed layer-wise inference.
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers
        # are repeated. Therefore, we compute the representation of all nodes
        # layer by layer.  The nodes on each layer are of course splitted in
        # batches.
        # TODO: can we standardize this?
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.n_hidden),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([10],output_device='cpu')
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                # print('rank: ',g.rank(),' ',blocks)
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                # print('rank:',g.rank(),' ',h_dst,' ',len(h_dst))
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    # h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield


class GraphSAGE(nn.Module):

    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    # def forward(self, blocks, x):
    #     h = x
    #     for i, (layer, block) in enumerate(zip(self.layers, blocks)):
    #         h = layer(block, h)
    #         if i != len(self.layers) - 1:
    #             h = self.activation(h)
    #             h = self.dropout(h)
            
    #     return h

    def forward(self, mfgs, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i > 0:
                h = self.activation(h)
                h = self.dropout(h)
            # print(h[:mfgs[i].num_dst_nodes()].shape)
            h_dst = h[:mfgs[i].num_dst_nodes()]
            # h = self.layers[i](mfgs[i], (h, h_dst))
            h = layer(mfgs[i], (h, h_dst))
        return h

    def inference_test(self, g, x, batch_size, device,nodes_tmp,model):
        
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )

        y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
        y = th.ones((g.num_nodes(), self.n_classes),dtype=th.float32)
        print(f"|V|={nodes.shape[0]}, eval batch size: {batch_size}")
            # print('y:',y.shape)
        sampler = dgl.dataloading.NeighborSampler([10,10])
        # sampler = dgl.dataloading.NeighborSampler([4,4])
        # sampler = dgl.dataloading.NeighborSampler([-1,-1])
            # sampler = dgl.dataloading.NeighborSampler([10,25])
        dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes_tmp,
                sampler,
                batch_size=len(nodes_tmp),
                shuffle=False,
                drop_last=False,
            )
        


        for input_nodes, output_nodes, blocks in (dataloader):

                h = x[input_nodes].to(device)
                for i, layer in enumerate(self.layers):
                    block = blocks[i].to(device)
                    h_dst = h[: block.number_of_dst_nodes()]
                    print(block.number_of_dst_nodes(),' ',block.num_dst_nodes())
                    h = layer(block, (h, h_dst))
                    if i != len(self.layers) - 1:
                        h = self.activation(h)
                        h = self.dropout(h)

                y[output_nodes] = h.cpu()
        g.barrier()
        return y

    def inference(self, g, x, batch_size, device):
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.n_hidden),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([10],output_device='cpu')
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                # print('rank: ',g.rank(),' ',blocks)
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                # print('rank:',g.rank(),' ',h_dst,' ',len(h_dst))
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield



def compute_acc(pred, label,is_multilabel):
    """
    Compute the accuracy of prediction given the label.
    """
    label = label.long()
    if is_multilabel:
        # output = model(mfgs, inputs)
        # output = th.sigmoid(pred)
        output = th.where(pred > 0.5, 1, 0).cpu().numpy()
        f1_score = sklearn.metrics.f1_score(label.cpu(), output, average="micro")
        accuracy = f1_score
        return accuracy
    else:
        # predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        return (th.argmax(pred, dim=1) == label).float().sum() / len(pred)



def evaluate_test(model, g, inputs, label, val_nid, test_nid, batch_size, device,is_multilabel):
    model.eval()
    with th.no_grad():
        time_val_start = time.time()
        # print('val:',val_nid)
        all_val_acc = 0
        pred_val = model.inference_test(g, inputs, batch_size, device,val_nid,model)
        val_acc = compute_acc(pred_val[val_nid], label[val_nid],is_multilabel)
        val_time = time.time()-time_val_start
        if not is_multilabel:
            label_tmp = label[val_nid].long()
            val_right_num = (th.argmax(pred_val[val_nid], dim=1) == label_tmp).float().sum()
            val_sum_num = len(pred_val[val_nid])
            print('part: ',g.rank(),"test:",val_sum_num,val_right_num)
            tensor_0 = th.tensor([val_right_num.item(),val_sum_num])
            # print(g.rank(),val_right_num.item(),val_sum_num)
            tensor_tmp = th.zeros(1,2)
            tensor_sum = th.zeros(1,2)
            if g.rank() == 0:
                dist.recv(tensor=tensor_tmp,src=1)
                tensor_sum += tensor_tmp

                # dist.recv(tensor=tensor_tmp,src=2)
                # tensor_sum += tensor_tmp

                # dist.recv(tensor=tensor_tmp,src=3)
                # tensor_sum += tensor_tmp

                result_arry = (tensor_sum+tensor_0).numpy()
                print(result_arry)
                all_val_acc = result_arry[0][0]/result_arry[0][1]
                print('All val acc:',all_val_acc)
            else:
                dist.send(tensor=th.tensor([val_right_num.item(),val_sum_num]),dst=0)

        else:
            
            output = pred_val[val_nid]
            label_set = label[val_nid]
            if g.rank() == 0:
                gather_list = [th.zeros(2,dtype=th.int64) for _ in range(2)]
                test = th.zeros(1,5)
                test_1 = th.zeros([1,5])
                print(test.shape,' ',test_1.shape)
                size_tensor = th.tensor([output.shape[0],output.shape[1]],dtype=th.int64)
                print(size_tensor.shape)
                dist.gather(size_tensor, dst = 0, gather_list=gather_list)
                print('before gather',' Rank ', g.rank(), ' has data ', size_tensor)
                print('gather_list:', gather_list)
                
                pre_tensor_list = []
                label_tensor_list = []
                
                for tensor in gather_list:
                    # print(th.zeros([tensor[0],tensor[1]]).shape)
                    pre_tensor_list.append(th.zeros([tensor[0],tensor[1]],dtype=th.float32))
                    label_tensor_list.append(th.zeros([tensor[0],tensor[1]],dtype=th.int64))
                print(output.dtype,label_set.dtype)
                dist.gather(output, dst = 0, gather_list=pre_tensor_list)
                dist.gather(label_set, dst = 0, gather_list=label_tensor_list)
                for i in range(len(pre_tensor_list)-1):
                    output = th.cat([output,pre_tensor_list[i+1]])
                    label_set = th.cat([label_set,label_tensor_list[i+1]])
                output = th.where(output > 0.5, 1, 0).cpu().numpy()
                f1_score = sklearn.metrics.f1_score(label_set.cpu(), output, average="micro")
                all_val_acc = f1_score
            else:
                dist.gather(tensor=th.tensor([output.shape[0],output.shape[1]],dtype=th.int64),dst=0)
                print('before gather',' Rank ', g.rank(), ' has data ', th.tensor([output.shape[0],output.shape[1]]))
                dist.gather(tensor=output,dst=0)
                dist.gather(tensor=label_set,dst=0)
               
    model.train()
    if g.rank() ==0:
        return all_val_acc, val_acc,  val_time
    else:
        return val_acc,  val_time


def evaluate(model, g, inputs, label, val_nid, test_nid, batch_size, device,is_multilabel):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The feat of all the nodes.
    label : The label of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        time_val_start = time.time()
        # inputs = inputs.to('cuda')
        pred = model.inference(g, inputs, batch_size, device)
        val_acc = compute_acc(pred[val_nid], label[val_nid],is_multilabel)
        all_val_acc = 0
        val_time = time.time()-time_val_start

        if not is_multilabel:
            label_tmp = label[val_nid].long()
            val_right_num = (th.argmax(pred[val_nid], dim=1) == label_tmp).float().sum()
            val_sum_num = len(pred[val_nid])
            print('part: ',g.rank(),"test:",val_sum_num,val_right_num)
            tensor_0 = th.tensor([val_right_num.item(),val_sum_num])
            # print(g.rank(),val_right_num.item(),val_sum_num)
            tensor_tmp = th.zeros(1,2)
            tensor_sum = th.zeros(1,2)
            if g.rank() == 0:
                dist.recv(tensor=tensor_tmp,src=1)
                tensor_sum += tensor_tmp

                # dist.recv(tensor=tensor_tmp,src=2)
                # tensor_sum += tensor_tmp

                # dist.recv(tensor=tensor_tmp,src=3)
                # tensor_sum += tensor_tmp

                result_arry = (tensor_sum+tensor_0).numpy()
                print(result_arry)
                all_val_acc = result_arry[0][0]/result_arry[0][1]
                print('All val acc:',all_val_acc)
            else:
                dist.send(tensor=th.tensor([val_right_num.item(),val_sum_num]),dst=0)

        else:
            
            output = pred[val_nid]
            label_set = label[val_nid]
            if g.rank() == 0:
                gather_list = [th.zeros(2,dtype=th.int64) for _ in range(2)]
                test = th.zeros(1,5)
                test_1 = th.zeros([1,5])
                print(test.shape,' ',test_1.shape)
                size_tensor = th.tensor([output.shape[0],output.shape[1]],dtype=th.int64)
                print(size_tensor.shape)
                dist.gather(size_tensor, dst = 0, gather_list=gather_list)
                print('before gather',' Rank ', g.rank(), ' has data ', size_tensor)
                print('gather_list:', gather_list)
                
                pre_tensor_list = []
                label_tensor_list = []
                
                for tensor in gather_list:
                    # print(th.zeros([tensor[0],tensor[1]]).shape)
                    pre_tensor_list.append(th.zeros([tensor[0],tensor[1]],dtype=th.float32))
                    label_tensor_list.append(th.zeros([tensor[0],tensor[1]],dtype=th.int64))
                print(output.dtype,label_set.dtype)
                dist.gather(output, dst = 0, gather_list=pre_tensor_list)
                dist.gather(label_set, dst = 0, gather_list=label_tensor_list)
                for i in range(len(pre_tensor_list)-1):
                    output = th.cat([output,pre_tensor_list[i+1]])
                    label_set = th.cat([label_set,label_tensor_list[i+1]])
                output = th.where(output > 0.5, 1, 0).cpu().numpy()
                f1_score = sklearn.metrics.f1_score(label_set.cpu(), output, average="micro")
                all_val_acc = f1_score
            else:
                dist.gather(tensor=th.tensor([output.shape[0],output.shape[1]],dtype=th.int64),dst=0)
                print('before gather',' Rank ', g.rank(), ' has data ', th.tensor([output.shape[0],output.shape[1]]))
                dist.gather(tensor=output,dst=0)
                dist.gather(tensor=label_set,dst=0)

    model.train()
    if g.rank() ==0:
        return all_val_acc, val_acc,  val_time
    else:
        return val_acc,  val_time


from torch.utils.data.distributed import DistributedSampler
def run(args, device, data, is_multilabel):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    shuffle = True
    # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet.
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
        # use_uva=True,
    )
    tmp_time = time.time()
    print(time.time()-tmp_time)
    # Define model and optimizer
    model = GCN(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    
    # model = Model(in_feats,args.num_hidden,n_classes)
    # model.
    tmp1_time =time.time()
    model = model.to(device)
    print('model time',time.time()-tmp1_time)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], output_device=device
            )
    
    # loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = F.binary_cross_entropy_with_logits()
    # loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # Training loop
    iter_tput = []
    epoch = 0
    train_time_list = []
    val_time_list = []
    test_time_list = []
    acc_tmp = 0
    epoch_tmp = 0
    counter = 0
    counter_test = 0
    run_time = time.time()
    for epoch in range(args.num_epochs):
        # sampler.set_epoch(epoch)
        tic = time.time()
        transfer_time = 0
        gather_time = 0
        comp_time = 0
        sample_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph
        # as a list of blocks.
        step_time = []
        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                comp_tmp = time.time()
                tic_step = time.time()
                sample_time += tic_step - start
                # fetch feat/label
       
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                # move to target device
                # print(device)
                
                batch_inputs, batch_labels,gather_time_tmp,transfer_time_tmp = load_subtensor(
                    g, seeds, input_nodes, device
                )
                gather_time +=gather_time_tmp
                tmp = time.time()
                blocks = [block.to(device) for block in blocks]
                
                # batch_inputs = batch_inputs.to('cuda')
                # batch_labels = batch_labels.to('cuda')
                
                transfer_time = transfer_time+transfer_time_tmp+time.time()-tmp
                # Compute loss and prediction
                start = time.time()
                # print(batch_inputs,blocks,batch_inputs.shape)
                batch_labels = batch_labels.long()
                batch_pred = model(blocks, batch_inputs)
                # batch_pred = model(g.ndata['feat'][seeds])
                # print(batch_pred.shape)
                # print(batch_pred.shape,batch_labels.shape)
                if is_multilabel:
                    # print(batch_labels.shape,'  ',batch_pred.shape)
                    # loss = F.binary_cross_entropy_with_logits(batch_pred, batch_labels.float())
                    loss = th.nn.BCEWithLogitsLoss(weight=None,reduction='sum')(batch_pred, batch_labels.float())
                else:
                    loss = F.cross_entropy(batch_pred, batch_labels)
                
                forward_end = time.time()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_time += forward_end - start
                backward_time += compute_end - forward_end
                
                optimizer.step()
                update_time += time.time() - compute_end
                comp_time += time.time()-comp_tmp
                
                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                if step % args.log_every == 0:
                    if is_multilabel:
                        # predictions = th.sigmoid(batch_pred)
                        predictions = th.where(batch_pred > 0.5, 1, 0).cpu().numpy()
                        # print('pre:',predictions[0:5][0:10])
                        # print('labels:',batch_labels[0:5][0:10])
                        f1_score = sklearn.metrics.f1_score(batch_labels.cpu(), predictions, average="micro")
                        acc = f1_score
                        # train_predictions.append(predictions)
                    else:
                        # train_predictions.append(predictions.argmax(1).cpu().numpy())
                        acc = compute_acc(batch_pred, batch_labels,is_multilabel)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )
                    print(
                        "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                        "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                        "{:.1f} MB | time {:.3f} s".format(
                            g.rank(),
                            epoch,
                            step,
                            loss.item(),
                            acc.item(),
                            np.mean(iter_tput[3:]),
                            gpu_mem_alloc,
                            np.sum(step_time[-args.log_every :]),
                        )
                    )
                start = time.time()

        toc = time.time()
        train_time_list.append(toc - tic)
        print(
            "Part {}, Epoch Time(s): {:.4f}, batch_pre: {:.4f} , gather: {:.4f} transfer: {:.4f}, compute: {:.4f} ,sample+data_copy: {:.4f}, "
            "forward: {:.4f}, backward: {:.4f},  update: {:.4f}, #seeds: {}, "
            "#inputs: {}".format(
                g.rank(),
                toc - tic,
                toc-tic-comp_time,
                gather_time,
                transfer_time,
                comp_time-transfer_time-gather_time,
                sample_time,
                forward_time,
                backward_time,
                update_time,
                num_seeds,
                num_inputs,
            )
        )
        epoch += 1


        if epoch % args.eval_every == 0 and epoch != 0:
            start = time.time()
            if g.rank()!=0:
                val_acc,  val_time= evaluate(
                    model if args.standalone else model.module,
                    g,
                    g.ndata["feat"],
                    g.ndata["label"],
                    val_nid,
                    test_nid,
                    args.batch_size_eval,
                    device,
                    is_multilabel,
                )
                print(
                    "Part {}, Val Acc {:.4f},  val_time: {:.4f}, ".format(
                        g.rank(), val_acc,  val_time 
                    )
                )
                val_time_list.append(val_time)
                # test_time_list.append(test_time)
            else:
                all_val_acc, val_acc,  val_time= evaluate(
                    model if args.standalone else model.module,
                    g,
                    g.ndata["feat"],
                    g.ndata["label"],
                    val_nid,
                    test_nid,
                    args.batch_size_eval,
                    device,
                    is_multilabel,
                )


        print('val test sample 2 blocks!!!!')

        if epoch % args.eval_every == 0 and epoch != 0:
            start = time.time()
            if g.rank()!=0:
                val_acc,  val_time= evaluate_test(
                    model if args.standalone else model.module,
                    g,
                    g.ndata["feat"],
                    g.ndata["label"],
                    val_nid,
                    test_nid,
                    args.batch_size_eval,
                    device,
                    is_multilabel,
                )
                print(
                    "Part {}, Test Val Acc {:.4f},  val_time: {:.4f}, ".format(
                        g.rank(), val_acc,  val_time 
                    )
                )
                val_time_list.append(val_time)
                # test_time_list.append(test_time)
            else:
                all_val_acc, val_acc,  val_time= evaluate_test(
                    model if args.standalone else model.module,
                    g,
                    g.ndata["feat"],
                    g.ndata["label"],
                    val_nid,
                    test_nid,
                    args.batch_size_eval,
                    device,
                    is_multilabel,
                )
                if all_val_acc > acc_tmp:
                    acc_tmp = all_val_acc
                    epoch_tmp = epoch
                    counter_test = 0
                else:
                    counter_test+=1

                print(
                    "Part {}, All Test Val ACC {:.4f} Test Val Acc {:.4f},  val_time: {:.4f},  run_time: {:.4f}".format(
                        g.rank(), all_val_acc ,val_acc,  val_time,  time.time()-run_time
                    )
                )
                val_time_list.append(val_time)
                # test_time_list.append(test_time)
                if counter_test == 10:
                    print('Run time:{},Best ALL VAL ACC:{} Epoch:{}'.format(time.time()-run_time,acc_tmp,epoch_tmp))
                    break


def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        th.distributed.init_process_group(backend=args.backend)
    print(socket.gethostname(), "Initializing DistGraph")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print(socket.gethostname(), "rank:", g.rank())

    pb = g.get_partition_book()
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=True
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=True
        )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    # print(g.rank(), len(local_nid))
    print(
        "part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
        "(local: {})".format(
            g.rank(),
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )
    del local_nid
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
        # device = th.device("cuda')
    n_classes = args.n_classes
    if n_classes == 0:
        label = g.ndata["label"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(label[th.logical_not(th.isnan(label))]))
        del label
    

    # Pack data
    in_feats = g.ndata["feat"].shape[1]
    
    # print(len(g.ndata['label'].shape))
    if len(g.ndata['label'].shape) > 1:
        is_multilabel = True
        num_classes = g.ndata['label'].shape[1]
        n_classes = num_classes
    else:
        is_multilabel = False
        label = g.ndata["label"][np.arange(g.num_nodes())]
        num_classes = len(th.unique(label[th.logical_not(th.isnan(label))]))
    # val_nid = th.tensor([0,1,2])    
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    print(val_nid)
    # val_nid = th.tensor([0,1,2])
    print("#label:", n_classes)
    run(args, device, data, is_multilabel)
    
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=6000)
    parser.add_argument("--batch_size_eval", type=int, default=6000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
        "of batches to be the same.",
    )
    args = parser.parse_args()
    
    print(args)
    main(args)