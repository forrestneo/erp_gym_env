import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

cash = 500
Total_loans = 0
Loan_balance = 0
Loan_interest = 0
Loan_interest_t = 0
Loan_rate_l = 0.1
Loan_rate_s = 0.06
Loan_term_l = 3
Loan_term_s = 1
R_price=10
R_ordert=1
R_store_c=1
L_price=50
L_rv=10
L_Change_cost=10
A_price=10
A_maintain=5
A_sales_v=1
P_price=25
P_leadtime=1
class ERP(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, ):
        super(ERP, self).__init__()
        #self.observation_t = np.zeros(12)
        self.cash = cash #初始化现金量
        self.cash_t = cash
        # self.Total_loans = Total_loans #初始化贷款总额
        # self.Loan_balance = Loan_balance #初始化贷款余额
        # self.Loan_interest = Loan_interest #初始化贷款总利息
        # self.Loan_interest_t = Loan_interest_t #初始化当期贷款利息
        #self.observation_space_n = len(self.observation_t)

        #self.action_space_n = 5
        self.action_space = spaces.Box(
            low=0, high=500, shape=(5, ), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12, ), dtype=np.float16)

        #贷款
        # self.Loan_rate_l = Loan_rate_l #初始化长期贷款利率
        self.Loan_rate_s = Loan_rate_s #初始化短期贷款利率
        # self.Loan_term_l = Loan_term_l #初始化长期贷款周期
        self.Loan_term_s = Loan_term_s #初始化短期贷款周期


        #原料
        self.R_price = R_price #原料单价
        self.R_ordert = R_ordert #原料提前预定时间
        self.R_store_c = R_store_c #原料储存价格


        #生产线
        self.L_price = L_price                      #生产线单价
        self.L_rv = L_rv                            #生产线残值
        self.L_Change_cost = L_Change_cost          #转产费用


        #销售区域
        self.A_price = A_price                      #销售区域单价
        self.A_maintain = A_maintain                #销售区域维护费
        self.A_sales_v = A_sales_v                  #销售区域销量


        #产品
        self.P_price = P_price                      #产品价格
        self.P_leadtime = P_leadtime                #产品生产周期

    def _take_action(self,action):
        loan = action[0]                            # 贷款金额
        pay_back = 0 #需还款金额
        loan_index = []
        for c,l in enumerate(self.loan_list):
            l[1]-=1
            if l[1] == 0:
                pay_back+=l[0]
                loan_index.append(c)
        for d in loan_index:
            del self.loan_list[d]
        self.loan_list.append([loan,self.Loan_term_s])

        r_store_cost = self.R_num*self.R_store_c
        r_arrive = 0 #抵达原料数

        r_buy = action[1] #购买原料数量
        r_cost = r_buy*self.R_price
        r_index = []
        r_way_num = 0
        for c,l in enumerate(self.R_list):
            l[1]-=1
            if l[1] == 0:
                r_arrive+=l[0]
                r_index.append(c)
            else:
                r_way_num += l[0]

        for d in r_index:
            del self.R_list[d]

        r_way = r_way_num * self.R_price  # 在途原料金额
        self.R_list.append([r_buy,self.R_ordert])

        self.R_num += r_arrive #原料库存

        loan_balance = self.observation_t[1] + loan - pay_back #贷款余额
        loan_all = self.observation_t[2] + loan #贷款总额
        loan_interest_t = self.observation_t[1]*self.Loan_rate_s #当期贷款利息
        loan_interest_all = self.observation_t[3]+loan_interest_t #贷款总利息


        l_decide = action[3] #新建或处置生产线数量
        l_cost = self.L_price*l_decide if l_decide > 0 else l_decide*self.L_Change_cost #新建或处置生产线费用/收入
        if self.L_num + l_decide <0:
            pass
        else:
            self.L_num += l_decide

        a_decide = action[4] #新建或处置销售区域数量
        a_cost = self.A_price*a_decide if a_decide>0 else 0 #开拓销售区域费用
        a_m_cost = self.A_num*self.A_maintain #销售区域维护费用
        if self.A_num + a_decide <0:
            pass
        else:
            self.A_num += a_decide

        p_pnums = action[2] #产品生产数量
        r_dq = p_pnums*1 #原料需求数量
        l_dq = p_pnums*1 #生产线需求数量

        p_index = []
        p_arrive = 0  # 当期产品增加量
        p_way_num = 0
        for c, l in enumerate(self.P_list):
            l[1] -= 1
            if l[1] == 0:
                p_arrive += l[0]
                p_index.append(c)
                self.L_num += 1
            else:
                p_way_num += l[0]
        for d in p_index:
            del self.P_list[d]
        p_way = p_way_num * self.P_price
        if r_dq > self.R_num:
            if l_dq > self.L_num:
                p_pnums = min(self.R_num,self.L_num) #实际生产数量
                self.R_num -= p_pnums
                self.L_num -= p_pnums
            else:
                p_pnums = self.R_num
                self.R_num -= p_pnums
                self.L_num -= p_pnums
        else:
            if l_dq > self.L_num:
                p_pnums = self.L_num
                self.R_num -= r_dq
                self.L_num -= p_pnums
            else:
                p_pnums = p_pnums
                self.R_num -= r_dq
                self.L_num -= p_pnums

        self.P_list.append([p_pnums, self.P_leadtime])
        self.P_num += p_arrive  # 产品库存

        sale_num = self.A_num*self.A_sales_v
        if sale_num > self.P_num:
            sale_num = self.P_num
        sales_revenue = sale_num * self.P_price
        self.cash = self.cash + loan - l_cost - a_cost - r_cost - loan_interest_t - r_store_cost -\
                            - a_m_cost + sales_revenue
        L_C = p_way + self.R_num #生产线产能
        Total_value = L_C*self.L_rv
        Total_sale = self.A_num*1
        self.observation_t = [self.cash,loan_balance,loan_all,loan_interest_all,loan_interest_t,
                              r_way,p_way,self.R_num,self.P_num,L_C,Total_sale,Total_value]

        rewrad = (self.cash + self.R_num*self.R_price +self.P_num*self.P_price+Total_value-\
                  (loan_balance+loan_interest_all+0-0))/self.cash_t
        state = self.observation_t
        return state,rewrad

    def step(self,action):
        done = False
        self.t += 1
        #state = self.observation_t.copy()
        obs,reward = self._take_action(action)
        reward = round(reward,4)
        if obs[0]<0 or self.t > 120:
            done = True
        return obs,reward,done,{}

    def reset(self):
        self.t = 0
        self.loan_list = []  # 贷款金额/还款时间
        self.R_num = 0
        self.R_list = []
        self.L_num = 0
        self.A_num = 0
        self.P_num = 0
        self.P_list = []
        self.observation_t = np.array([500,0,0,0,0,0,0,0,0,0,0,0])
        state = self.observation_t
        return state
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(self.obs,self.r)
