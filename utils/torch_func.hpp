#ifndef _TORCH_FUNC_
#define _TORCH_FUNC_
#include "torch/torch.h"
#include "core/neutronstar.hpp"

void test_torch_func() {
  NtsVar a = torch::ones({2, 4});
  std::cout << a << std::endl;
}

long get_correct(NtsVar &input, NtsVar &target, bool onelabel=true) {
  long ret = 0;
  if (!onelabel) {
    NtsVar predict = torch::where(torch::sigmoid(input) > 0.5, 1, 0);
    auto equal = predict == target;
    for (int i = 0; i < input.size(0); ++i) {
      ret += equal[i].all().item<int>();
    }
    // for (int i = 0; i < input.size(0); ++i) {
    //   auto tmp = predict[i].to(torch::kLong).eq(target[i]).to(torch::kLong);
    //   ret += tmp.all().item<int>();
    // }
  } else {
    NtsVar predict = input.argmax(1);
    NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
    ret = output.sum(0).item<long>();
  }
  return ret;
}


float f1_score(NtsVar &input, NtsVar &target, bool onelabel = true) {
  float ret;
  // if (graph->config->classes > 1) {
  if (!onelabel) {
    NtsVar predict = input.sigmoid();
    predict = torch::where(predict > 0.5, 1, 0);
    auto all_pre = predict.sum();
    auto x_tmp = torch::where(target == 0, 2, 1);
    auto true_p = (x_tmp == predict).sum();
    auto all_true = target.sum();
    auto precision = true_p / all_pre;
    auto recall = true_p / all_true;
    auto f2 = 2 * precision * recall / (precision + recall);
    ret =  f2.item<float>();
  } else {
    NtsVar predict = input.argmax(1);
    NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
    ret = output.sum(0).item<long>();
  }
  return ret;
}  

NtsVar Loss(NtsVar &output, NtsVar &target, bool onelabel=true) {
  // std::cout << "start loss" << std::endl;
  NtsVar loss; 
  if (!onelabel) {
    // loss = torch::binary_cross_entropy_with_logits(output, target.to(torch::kFloat));
    loss = torch::binary_cross_entropy(torch::sigmoid(output), target.to(torch::kFloat));

  } else {
    torch::Tensor a = output.log_softmax(1);
    loss = torch::nll_loss(a, target);
  }
  // std::cout << "loss " << loss.item<float>() << std::endl;
  // std::cout << "loss in loop " << loss.data_ptr() << " " << loss.item<float>() << std::endl;
  return loss;
}

// void Test(long s) { // 0 train, //1 eval //2 test
//   NtsVar mask_train = MASK.eq(s);
//   NtsVar all_train =
//       X[graph->gnnctx->layer_size.size() - 1]
//           .argmax(1)
//           .to(torch::kLong)
//           .eq(L_GT_C)
//           .to(torch::kLong)
//           .masked_select(mask_train.view({mask_train.size(0)}));
//   NtsVar all = all_train.sum(0);
//   long *p_correct = all.data_ptr<long>();
//   long g_correct = 0;
//   long p_train = all_train.size(0);
//   long g_train = 0;
//   MPI_Datatype dt = get_mpi_data_type<long>();
//   MPI_Allreduce(p_correct, &g_correct, 1, dt, MPI_SUM, MPI_COMM_WORLD);
//   MPI_Allreduce(&p_train, &g_train, 1, dt, MPI_SUM, MPI_COMM_WORLD);
//   float acc_train = 0.0;
//   if (g_train > 0)
//     acc_train = float(g_correct) / g_train;
//   if (graph->partition_id == 0) {
//     if (s == 0) {
//       LOG_INFO("Train Acc: %f %d %d", acc_train, g_train, g_correct);
//     } else if (s == 1) {
//       LOG_INFO("Eval Acc: %f %d %d", acc_train, g_train, g_correct);
//     } else if (s == 2) {
//       LOG_INFO("Test Acc: %f %d %d", acc_train, g_train, g_correct);
//     }
//   }
// }

#endif