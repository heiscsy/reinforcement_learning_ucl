#include <vector>
#include <assert.h>
#include <cmath>
#include <memory>
#include <iostream>
#include <limits>

namespace {

using ValueType = std::vector<std::vector<double> >;
using PolicyType = std::vector<std::vector<std::vector<double> > >; 

std::vector<double> Fac(21, 0.0);
std::vector<double> Request1(21, 0.0);
std::vector<double> Request2(21, 0.0);
std::vector<double> Return1(21, 0.0);
std::vector<double> Return2(21, 0.0);

void CalcFac() {
  Fac[0] = 1;
  Fac[1] = 1;
  for(auto i = 2; i<21; ++i) {
    Fac[i] = i * Fac[i-1];
  }
}

double diff(const ValueType& value1, const ValueType& value2) {
  double ret = 0;
  for(size_t site1 = 0; site1<21; ++site1)
    for(size_t site2 = 0; site2<21; ++site2) {
      ret+=std::abs(value1[site1][site2] - value2[site1][site2]);
    }
  return ret;
}

double diff(const PolicyType& policy1, const PolicyType& policy2) {
  double ret = 0;
  for(size_t site1 = 0; site1<21; ++site1)
    for(size_t site2 = 0; site2<21; ++site2) 
      for(size_t action = 0; action<11; ++action){
      ret+=std::abs(policy1[site1][site2][action] 
        - policy2[site1][site2][action]);
    }
  return ret;
}

double Poisson(double lam, int number) {
  return std::exp(-lam) * (std::pow(lam, number)) / (
    Fac[number]+std::numeric_limits<double>::min());
}

void CalcProbTable() {
  for (int i =0; i<21; ++i) {
    Request1[i] = Poisson(3, i);
    Request2[i] = Poisson(4, i);
    Return1[i] = Poisson(3, i);
    Return2[i] = Poisson(2, i);
  }
}

void ValueEstimation(const ValueType& value, const PolicyType& policy, 
    ValueType* value_new) {
  value_new->resize(21);
  for (auto& v: *value_new) {
    v.resize(21, 0.0);
  }
  for (auto site1 = 0; site1<21; site1++) 
    for (auto site2 = 0; site2<21; site2++) {
      auto max_move = std::min(std::min(site1, 5), 20-site2);
      auto max_receive = std::min(std::min(site2, 5), 20-site1);
      for (auto action=-max_move; action<max_receive+1; ++action) {
        auto prob_policy = policy[site1][site2][action+5];
        auto site1_remain = site1 + action;
        auto site2_remain = site2 - action;
        auto reward = 0.0, expect_future_return = 0.0;
        for(auto site1_rent=0; site1_rent<site1_remain+1; ++site1_rent)
          for(auto site2_rent=0; site2_rent<site2_remain+1; ++site2_rent) {
            for(auto site1_return=0; site1_return<20-(site1_remain-site1_rent)+1; 
              ++site1_return)
              for(auto site2_return=0; site2_return<20-(site2_remain-site2_rent)+1;
                ++site2_return) {
                auto prob = Return1[site1_return] * Return2[site2_return] * 
                         Request1[site1_rent] * Request2[site2_rent];
                expect_future_return = expect_future_return + prob * value[
                    site1_return+site1_remain-site1_rent][site2_return+
                    site2_remain-site2_rent];
              }
            auto prob_rent = Request1[site1_rent] * Request2[site2_rent];
            // std::cout<<prob_rent<<std::endl;
            reward = reward + prob_rent *(site1_rent*10.0 + site2_rent*10.0 
            - 2*std::abs(action));
            //std::cout<<reward<<std::endl;
          }
        value_new->at(site1)[site2] = prob_policy*(reward + expect_future_return*0.9) 
          +value_new->at(site1)[site2];
      }
    }
}

void UpdatePolicy(const ValueType& value, PolicyType* policy_new) {
  policy_new->resize(21);
  for (auto& s2: *policy_new) {
    s2.resize(21);
    for (auto& p: s2) {
      p.resize(11, 0);
    }
  }
  for (auto site1 = 0; site1<21; site1++) 
    for (auto site2 = 0; site2<21; site2++) {
      auto max_move = std::min(std::min(site1, 5), 20-site2);
      auto max_receive = std::min(std::min(site2, 5), 20-site1);
      std::vector<double> action_list(11, 0.0);
      double max_expect = 0;
      size_t max_idx = 0;
      for (auto action=-max_move; action<max_receive+1; ++action) {
        auto site1_remain = site1 + action;
        auto site2_remain = site2 - action;
        auto reward = 0.0, expect_future_return = 0.0;
        for(auto site1_rent=0; site1_rent<site1_remain+1; ++site1_rent)
          for(auto site2_rent=0; site2_rent<site2_remain+1; ++site2_rent) {
            for(auto site1_return=0; site1_return<20-(site1_remain-site1_rent)+1; 
              ++site1_return)
              for(auto site2_return=0; site2_return<20-(site2_remain-site2_rent)+1;
                ++site2_return) {
                auto prob = Return1[site1_return] * Return2[site2_return] * 
                         Request1[site1_rent] * Request2[site2_rent];
                expect_future_return = expect_future_return + prob * value[
                    site1_return+site1_remain-site1_rent][site2_return+
                    site2_remain-site2_rent];
              }
          }
        if(expect_future_return > max_expect) {
          max_expect = expect_future_return;
          max_idx = action+5;
        }
      }
      action_list[max_idx] = 1.0;
      policy_new->at(site1)[site2] = action_list;
    }
}

void VisualizeValue(const ValueType& value) {
  for(auto site1 = 0; site1<21; ++site1) {
    for(auto site2 = 0; site2<21; ++site2) {
      std::cout<<value[site1][site2]<<" ";
    }
    std::cout<<std::endl;
  }
}

void VisualizePolicy(const PolicyType& policy) {
  for(auto site1 = 0; site1<21; ++site1) {
    for(auto site2 = 0; site2<21; ++site2) {
      for(auto action=0; action<11; ++action)
        std::cout<<policy[site1][site2][action]<<" ";
      std::cout<<std::endl;
    }
  }
}


void main() {
  CalcFac();
  CalcProbTable();
  ValueType value(21);
  for (auto& v: value) {
    v.resize(21, 0.0);
  }
  PolicyType policy(21);
  for (auto& s2: policy) {
    s2.resize(21);
    for (auto& p: s2) {
      p.resize(11, 1/11.0);
    }
  }

  ValueType value_new;
  while(1) {
    while(1) {
      ValueEstimation(value, policy, &value_new);
      if(diff(value_new, value)<10) break;
      value = std::move(value_new);
    }
    VisualizeValue(value_new);
    std::cout<<std::endl;
    PolicyType policy_new;
    UpdatePolicy(value, &policy_new);
    if(diff(policy_new, policy)<1) break;
    policy = std::move(policy_new);
    VisualizePolicy(policy);
    std::cout<<std::endl;
  }
}

}  // namespace 

int main(int argc, char** argv) {
  main();
  return 1;
}