#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector> 
#include <climits>
#include <cassert>
#include <thread>
#include <algorithm>
// #include <omp.h>
// #include <numa.h>
const int N = 1000000;


uint64_t get_time() { 
  return std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::steady_clock::now().time_since_epoch()).count();
}  


int read_edgelist(std::string input_dir, std::string dataname, std::vector<std::pair<int,int>> &edges) {
  std::string input_file = input_dir + '/' + dataname + ".edge";
  std::ifstream inFile(input_file, std::ios::in | std::ios::binary);
  if (!inFile.is_open()) {
    std::cout << input_file << " not exist!" << std::endl;
    exit(-1);
  }
  
  int u, v;
  int count = 0;
  int max_node_id = -1;
  int min_node_id = INT_MAX;
  while (inFile.read(reinterpret_cast<char*>(&u), sizeof(int))) {
    count++;
    inFile.read(reinterpret_cast<char*>(&v), sizeof(int));
    edges.push_back({u, v});
    max_node_id = std::max(max_node_id, std::max(u, v));
    min_node_id = std::min(min_node_id, std::min(u, v));
  }
  inFile.close();
  assert(min_node_id == 0);
  std::cout << "read " << count << " edges " << max_node_id + 1 << " nodes" << std::endl;
  return max_node_id + 1;
}


void read_mask(std::string input_dir, std::string dataname, std::vector<int> &node_mask) {
  std::string input_file = input_dir + '/' + dataname + ".mask";
  std::ifstream inFile(input_file, std::ios::in);
  if (!inFile.is_open()) {
    std::cout << input_file << " not exist!" << std::endl;
    exit(-1);
  }
  
  int id;
  std::string msk;
  int count = 0;
  while (inFile >> id) {
    inFile >> msk;
    count++;
    if (msk.compare("train") == 0) {
      node_mask[id] = 0;
    } else if (msk.compare("eval") == 0 || msk.compare("val") == 0) {
      node_mask[id] = 1;
    } else if (msk.compare("test") == 0) {
      node_mask[id] = 2;
    } else {
      node_mask[id] = 3;
    }
  }

  inFile.close();
  std::cout << "read " << count << " line " << std::endl;;
  return;
}


void bfs(int start_node, std::vector<std::vector<int>>& inG, int hop, std::vector<std::pair<int, uint64_t>> &timestamp) {
  timestamp.push_back({start_node, 0});
  std::vector<int> layer_nodes;
  layer_nodes.push_back(start_node);
  // std::unorder_set<int> visit;
  // TODO miletime
  // uint64_t start_time = time(nullptr);
  uint64_t start_time = get_time();
  
  // std::cout << "start time " << start_time << std::endl;

  // int max_node_id = start_node;
  for (int i = 0; i < hop; ++i) {
    std::unordered_set<int> curr_layer;
    for (auto &u : layer_nodes) {
      curr_layer.insert(inG[u].begin(), inG[u].end());
      for (auto &v : inG[u]) {
        // max_node_id = std::max(max_node_id, v);
        // timestamp.push_back({v, time(nullptr) - start_time});
        timestamp.push_back({v, get_time() - start_time});
        // if (timestamp.back().second == 0) {
        //   std::cout << "zero" << std::endl;
        // }
        // std::cout << get_time() - start_time << std::endl;
        
      }
    }
    layer_nodes.clear();
    std::copy(curr_layer.begin(), curr_layer.end(), std::back_inserter(layer_nodes));
  }
  // std::vector<uint64_t> unique_timestamp(max_node_id, UINT_MAX);
  // for (auto& p : timestamp) {
  //   unique_timestamp[p->first] = std::min(unique_timestamp[p->first], p->second);
  // }
  return;
}

void do_bfs(int start_offset, int end_offset, std::vector<int>& all_nids, std::vector<std::vector<int>>& inG, int hop, std::vector<std::vector<std::pair<int, uint64_t>>>& all_pair) {
  // std::cout << "runing... " << start_offset << " " << end_offset << " " << std::endl;
  for (int i = start_offset; i < end_offset; ++i) {
    int curr_nid = all_nids[i];
    // std::cout << "start bfs " << curr_nid << std::endl;
    bfs(curr_nid, inG, hop, all_pair[i]);
  }
  // std::cout << "done..." << std::endl;
}


void do_bfs1(int start_offset, int end_offset) {
  for (int i = start_offset; i < end_offset; ++i) {
  }
}

auto bfs(int start_node, std::vector<std::vector<int>>& inG, int hop) {
  std::vector<std::pair<int,uint64_t>> timestamp;
  timestamp.push_back({start_node, 0});
  std::vector<int> layer_nodes;
  layer_nodes.push_back(start_node);
  // std::unorder_set<int> visit;
  uint64_t start_time = time(nullptr);


  int max_node_id = start_node;
  for (int i = 0; i < hop; ++i) {
    std::unordered_set<int> curr_layer;
    for (auto &u : layer_nodes) {
      curr_layer.insert(inG[u].begin(), inG[u].end());
      for (auto &v : inG[u]) {
        max_node_id = std::max(max_node_id, v);
        timestamp.push_back({v, time(nullptr) - start_time});
        // if (timestamp.back().second == 0) {
        //   std::cout << "zero" << std::endl;
        // }
      }
    }
    layer_nodes.clear();
    std::copy(curr_layer.begin(), curr_layer.end(), std::back_inserter(layer_nodes));
  }

  // std::vector<uint64_t> unique_timestamp(max_node_id, UINT_MAX);
  // for (auto& p : timestamp) {
  //   unique_timestamp[p->first] = std::min(unique_timestamp[p->first], p->second);
  // }
  return std::move(timestamp);
}  



double cross_edges(std::vector<std::vector<int>>& all_blocks, int block_id, int part_id, std::vector<std::vector<int>>& inG, std::vector<std::unordered_set<int>>& partition_nodes) {
  double count = 0;
  for (const auto& u : all_blocks[block_id]) {
    for (const auto& v : inG[u]) {
      if (partition_nodes[part_id].find(v) != partition_nodes[part_id].end()) {
        count++;
      }
    }
  }
  return count;
}



int main(int argc, char **argv) {

  if (argc < 6) {
    printf("Usage: ./bytegnn input_dir dataname num_parts num_hops out_dir\n");
    exit(-1);
  }
  
  std::vector<std::pair<int,int>> edges;
  std::string dataname = argv[2];
  int num_parts = std::stoi(argv[3]);
  int num_hops = std::stoi(argv[4]);
  std::string out_dir = argv[5];

  std::cout << "#################################" << std::endl;
  std::cout << "input directory: " << argv[1] << std::endl;
  std::cout << "dataset name: " << dataname << std::endl;
  std::cout << "num_parts: " << argv[3] << std::endl;
  std::cout << "num_hops: " << argv[4] << std::endl;
  std::cout << "output directory: " << out_dir << std::endl;
  std::cout << "#################################" << std::endl;


  int num_nodes = read_edgelist(argv[1], dataname, edges);


  std::vector<int> node_mask(num_nodes);
  read_mask(argv[1], argv[2], node_mask);


  // store in edges
  std::vector<std::vector<int>> inG(num_nodes);
  for (auto& p : edges) {
    inG[p.second].push_back(p.first);
  }
  edges.clear();
  
  std::vector<int> train_nids;
  std::vector<int> val_nids;
  std::vector<int> test_nids;
  std::vector<int> all_nids;
  for (int i = 0; i < num_nodes; ++i) {
    if (node_mask[i] == 0) {
      train_nids.push_back(i);
    } else if(node_mask[i] == 1) {
      val_nids.push_back(i);
    } else if (node_mask[i] == 2) {
      test_nids.push_back(i);
    }
  }
  node_mask.clear();
  std::copy(train_nids.begin(), train_nids.end(), std::back_inserter(all_nids));
  std::copy(val_nids.begin(), val_nids.end(), std::back_inserter(all_nids));
  std::copy(test_nids.begin(), test_nids.end(), std::back_inserter(all_nids));
  int num_mask = all_nids.size();
  int train_num = train_nids.size();  
  int test_num = test_nids.size();  
  int val_num = val_nids.size();  
  std::cout << "train " << train_num << ", val " << val_num << ", test " << test_num << ", all " << all_nids.size() << std::endl;

  std::unordered_set<int> train_st;
  std::unordered_set<int> val_st;
  std::unordered_set<int> test_st;
  train_st.insert(train_nids.begin(), train_nids.end());
  val_st.insert(val_nids.begin(), val_nids.end());
  test_st.insert(test_nids.begin(), test_nids.end());

  train_nids.clear();
  val_nids.clear();
  test_nids.clear();
  
  // std::vector<std::vector<std::pair<int, uint64_t>>> all_pair;
  // for (auto &u : all_nids) {
  //   all_pair.push_back(std::move(bfs(u, inG, 2)));
  // }

  // int num_thread = numa_num_configured_cpus();
  int num_thread = std::thread::hardware_concurrency();

  std::cout << "thread num " << num_thread << std::endl;

  // int count_start_time = time(nullptr);
  uint64_t count_start_time = get_time();
  std::vector<std::vector<std::pair<int, uint64_t>>> all_pair(all_nids.size());

  // for (int i = 0; i < num_mask; ++i) {
  //   std::cout << all_pair[i].size() << " ";
  // } std::cout << std::endl;

  std::vector<std::thread> thread_vec;
  for (int i = 0; i < num_thread; ++i) {
    int start_offset = num_mask / num_thread * i;
    int end_offset = i == num_thread - 1 ? num_mask : num_mask / num_thread * (i + 1);
    // do_bfs(start_offset, end_offset, all_nids, inG, 2, all_pair);
    thread_vec.push_back(std::thread(do_bfs, start_offset, end_offset, std::ref(all_nids), std::ref(inG), num_hops, std::ref(all_pair)));
    // thread_vec.push_back(std::thread(do_bfs1, start_offset, end_offset));
  }
  std::cout << "thread create done!" << std::endl;

  for (auto& tid : thread_vec) {
    tid.join();
  }

  // for (int i = 0; i < num_mask; ++i) {
  //   std::cout << all_pair[i].size() << " ";
  // } std::cout << std::endl;

  std::cout << "thread joined!" << std::endl;


  // for (int i = 0; i < num_mask; ++i) {
  //   int curr_nid = all_nids[i];
  //   bfs(curr_nid, inG, 2, all_pair[i]);
  // }
  // assert (all_nids.size() == all_pair.size());

  
  // int count_end_time = time(nullptr);
  uint64_t count_end_time = get_time();
  
  std::cout << "bfs cost " << (count_end_time - count_start_time) / 1e9 << "s"<< std::endl;


  // get block
  std::vector<uint64_t> min_timestamp(num_nodes, UINT64_MAX);
  std::vector<int> belongs(num_nodes, -1);

  // std::unordered_set<int> unique_node;
  for (int i = 0; i < all_nids.size(); ++i) {
    int curr_node = all_nids[i];
    // std::cout << i << " " << curr_node << " " << all_pair[i].size() << std::endl;
    for (auto& p : all_pair[i]) {
      // unique_node.insert(p.first);
      if (min_timestamp[p.first] <= p.second) continue;
      min_timestamp[p.first] = p.second;
      belongs[p.first] = curr_node;
      // unique_node.insert(p.first);
      // if (p.first == 169279) {
      //   std::cout << p.second << " " << belongs[p.first] << " " << curr_node <<  std::endl;
      // }
    }
  }
  min_timestamp.clear();

  // std::cout << "unique node " << unique_node.size() << std::endl; 

  for (auto u : all_nids) {
    // if (belongs[u] != u) {
      // std::cout << "node " << u << " belongs " << belongs[u] << " min_stamp " << min_timestamp[u] <<  std::endl;
    // }
    assert(belongs[u] == u);
  }

  all_nids.clear();


  int all_active_nodes = 0;
  std::vector<std::vector<int>> all_blocks(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    int u = belongs[i];
    if (u != -1) {
      all_active_nodes++;
      all_blocks[u].push_back(i);
    }
  }
  std::cout << "all_active_nodes " << all_active_nodes << std::endl;

  
  // all block size
  // for (auto & p : all_blocks) {
  //   std::cout << p.size() << " ";
  // } std::cout << std::endl;

  std::sort(all_blocks.begin(), all_blocks.end(), [](const auto& x, const auto &y) {
    return x.size() > y.size();
  });

  /////////// test sort
  for (int i = 1; i < all_blocks.size(); ++i) {
    assert(all_blocks[i].size() <= all_blocks[i - 1].size());
  }
  for (int i = 0; i < num_mask; ++i) {
    assert(all_blocks[i].size() >= 1);
  }
  for (int i = num_mask; i < all_blocks.size(); ++i) {
    assert(all_blocks[i].size() == 0);
  }
  ///////////



  // // all block size
  // for (auto & p : all_blocks) {
  //   std::cout << p.size() << " ";
  // } std::cout << std::endl;




  // assign blocks
  double alpha = 1.0;
  double beta = 1.0;
  double gamma = 1.0;
  std::vector<std::unordered_set<int>> partition_nodes(num_parts);
  std::vector<std::vector<int>> partition_mask_nodes(num_parts, std::vector<int>(3, 0));
  std::vector<int> parts_result(num_nodes, -1);

  for (int i = 0; i < num_mask; ++i) {
    std::vector<double> score(num_parts, 0);
    for (int j = 0; j < num_parts; ++j) {
      double ce = 1;
      if (partition_nodes[j].size() > 0) {
        ce += cross_edges(all_blocks, i, j, inG, partition_nodes) / partition_nodes[j].size();
      }
      int p_train = partition_mask_nodes[j][0];
      int p_val = partition_mask_nodes[j][1];
      int p_test = partition_mask_nodes[j][2];
      double bs = 1 - alpha * p_train / train_num - beta * p_val / val_num - gamma * p_test / test_num;
      score[j] = ce * bs;
    }
    int idx = 0;
    for (int j = 0; j < num_parts; ++j) {
      if (score[j] > score[idx]) {
        idx = j;
      }
    }
    for (const auto& u : all_blocks[i]) {
      parts_result[u] = idx;
      // udpate partition mask nodes
      if (train_st.find(u) != train_st.end()) partition_mask_nodes[idx][0]++;
      if (val_st.find(u) != val_st.end()) partition_mask_nodes[idx][1]++;
      if (test_st.find(u) != test_st.end()) partition_mask_nodes[idx][2]++;
      
    }
    // update partition nodes
    partition_nodes[idx].insert(all_blocks[i].begin(), all_blocks[i].end());
  }

  //  release memory
  all_blocks.clear();

  ////////// test assign blocks
  for (int i = 0; i < num_parts; ++i) {
    for (const auto& u : partition_nodes[i]) {
      assert(parts_result[u] == i);
    }
  }
  int test_assign_nodes = 0;
  for (int i = 0; i < num_nodes; ++i) {
    test_assign_nodes += parts_result[i] != -1;
  }
  assert(test_assign_nodes == all_active_nodes);
  ////////////////////


  // save partiton result
  // std::string save_partition_result = "~/neutron-sanzo/exp/Partition/partition/partition_result/bytegnn/" + dataname + "-parts.txt";
  std::string save_partition_result = out_dir + '/' + dataname + "-parts.txt";
  std::ofstream outFile(save_partition_result, std::ios::out);
  if (!outFile.is_open()) {
    std::cout << save_partition_result << " create failed!" << std::endl;
    exit(-1);
  }

  for (int i = 0; i < num_parts; ++i) {
    outFile << i << " " << partition_nodes[i].size();
    for (const auto& u : partition_nodes[i]) {
      outFile << " " << u;
    }
    outFile << std::endl;
  }

  return 0;
}
