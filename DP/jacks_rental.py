from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import time

kSite1Request = 3
kSite1Return = 3
kSite2Request = 4
kSite2Return = 2
kDiscount = 0.9

Request1 = np.zeros([21])
Request2 = np.zeros([21])
Return1 = np.zeros([21])
Return2 = np.zeros([21])
# rental request:
def GetRentalRequest():
  request1 = np.random.poisson(kSite1Request, 1)
  return1 = np.random.poisson(kSite1Return, 1)
  request2 = np.random.poisson(kSite2Request, 1)
  return2 = np.random.poisson(kSite2Return, 1)
  return request1, return1, request2, return2

def calcProbTable():
  for action in range(21):
    Request1[action] = Poisson(kSite1Request, action)
    Request2[action] = Poisson(kSite2Request, action)
    Return1[action] = Poisson(kSite1Return, action)
    Return2[action] = Poisson(kSite2Return, action)

def Poisson(lam, number):
  return np.exp(-lam)*(lam ** number) / np.math.factorial(number)

def ValueEstimation(value, policy):
  value_new = np.zeros([21, 21])
  for site1 in range(21):
    for site2 in range(21):
      # bellman expection backup 
      # v'(s) = E(R + v(s(t+1)))
      # action = [-5,5]
      max_move = np.min([site1, 5, 20-site2])
      max_receive = np.min([site2, 5, 20-site1])
      possible_action = range(-max_move, max_receive+1)
      for action in possible_action:
        prob_policy = policy[site1, site2, action+5]
        site1_remain = site1 + action
        site2_remain = site2 - action
        reward = 0
        expect_future_return = 0
        for site1_rent in range(site1_remain+1):
          for site2_rent in range(site2_remain+1):
            for site1_return in range(20-(site1_remain - site1_rent)+1):
              for site2_return in range(20-(site2_remain - site2_rent)+1):
                prob = Return1[site1_return] * Return2[site2_return] *\
                   Request1[site1_rent] * Request2[site2_rent]
                expect_future_return = expect_future_return + prob*\
                  value[site1_return+site1_remain-site1_rent, \
                   site2_return+site2_remain-site2_rent]
            prob_rent = Request1[site1_rent] * Request2[site2_rent]
            reward = reward + prob_rent*(site1_rent*10 + site2_rent*10 - \
              2*np.abs(action))
        value_new[site1,site2] = prob_policy*(reward + 
          expect_future_return * kDiscount) + \
          value_new[site1, site2]
  return value_new

def UpdatePolicy(value):
  policy_new = np.zeros([21,21,11])
  for site1 in range(21):
    for site2 in range(21):
      max_move = np.min([site1, 5, 20-site2])
      max_receive = np.min([site2, 5, 20-site1])
      possible_action = range(-max_move, max_receive+1)
      action_list = np.zeros([11])
      for action in possible_action:
        site1_remain = site1 + action
        site2_remain = site2 - action
        expect_future_return = 0
        for site1_rent in range(site1_remain+1):
          for site2_rent in range(site2_remain+1):
            for site1_return in range(20-(site1_remain - site1_rent)+1):
              for site2_return in range(20-(site2_remain - site2_rent)+1):
                prob = Return1[site1_return] * Return2[site2_return]*\
                  Request1[site1_rent] * Request2[site2_rent]
                expect_future_return = expect_future_return + prob*\
                  value[site1_return+site1_remain-site1_rent, \
                   site2_return+site2_remain-site2_rent]
        action_list[action+5] = expect_future_return
      idx = np.argmax(action_list)
      print(idx)
      policy_new[site1, site2, idx] = 1.0
    return policy_new

def main():
  calcProbTable()
  value = np.zeros([21, 21])
  policy = np.ones([21, 21, 11]) * (1 / 11)
  while True:
    while True:
      value_new = ValueEstimation(value, policy)
      if np.sum(np.abs(value_new - value))<100:
        break
      value = value_new
    print("value")
    print(value)
    policy_new = UpdatePolicy(value)
    if np.sum(np.abs(policy_new - policy))<10:
      break
    policy = policy_new
    print("policy")
    print(policy)
    print("====================")

if __name__=='__main__':
  main()  