import time
import numpy as np
import random
from random import randint

from numpy.matrixlib.defmatrix import matrix_power
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
import math
import pandas
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import timeit
from ortools.algorithms import pywrapknapsack_solver

'''Agent'''
class Player(Agent):
    def __init__(self, no, model, state, loc=(0,0),r=1):
        super().__init__(no, model)
        self.no=no
        # in the begin of simulation
        if r==1:
            age=randint(20, 36)
            self.age=age
        else:
            age=20
            self.age=20
        if loc==(0,0):
            location=(randint(1,7),randint(1,7))
        else:
            location=loc
        self.nationality=location
        self.location=location
        ability=math.floor(np.random.normal(80, 10, size=1)[0]-5*abs(age-28))
        if ability<0:
            self.ability=0
        else:
            self.ability=ability
        # the player is employed or not
        self.empolyed=state
        self.wage=0
        self.rank=0
    def step(self):
        self.age+=1
        ability=math.floor((self.ability+(np.random.normal(80, 20, size=1)[0]-5*abs(self.age-28)))/2)
        if ability<0:
            self.ability=0
        else:
            self.ability=ability
    def release(self):
        self.empolyed=False
    

class Team(Agent):
    def __init__(self, unique_id, model, pos, k, r,bonds=[]):
        super().__init__(unique_id, model)
        self.pos=pos
        self.member=[]
        self.member_ability=[]
        resource=r
        for j in range(k):
            resource*=0.8
        resource=int(resource)
        self.resource=resource
        self.performance=resource
        self.salarybudget=resource
        self.osalarybudget=resource
        self.wagesum=resource
        self.owagesum=resource
        self.performance2=resource
        self.entropy=0
        self.pnation=[]
        self.distance=[7,0,0,0,0,0]
        self.gridnumber=[0,0,0,0,0,0]
        self.nordistance=[7,0,0,0,0,0]
        self.nation=[0,0,0,0,0]
        self.bondgrid=bonds
        self.hirecost=[0,0,0,0,0,0,0]
        self.hireability=[0,0,0,0,0,0,0]
        

    
    def fire(self):
        # random release 7 player in the team
        n=len(self.member)
        self.member_rank=rank(self.member_ability)
        renew_member=[]
        renew_member_ability=[]
        if n>=21:
            r=random.sample(range(0, n),n-21)
            for i in r:
                self.member[i].release()
        for i in range(n):
            if self.member[i].empolyed==True:
                renew_member.append(self.member[i])
                renew_member_ability.append(self.member[i].ability)
        self.member=renew_member
        self.member_ability=renew_member_ability
                
            
    def init_recruit(self, model , n, loc):
        possible_player=[]
        new_player=[]
        for agent in model.p_schedule.agents:
            if type(agent) == Player and agent.empolyed==False and agent.age<=36 and agent.location==loc:
                possible_player.append(agent)
        random.shuffle(possible_player)
        new_player=possible_player[0:n]
        for player in new_player:
            self.model.grid.move_agent(player, self.pos)
            player.location=self.pos
            player.wage=player.ability
            player.empolyed=True
            self.member.append(player)
            self.member_ability.append(player.ability)
        self.cal_entropy()
        self.performance=sum(self.member_ability)
        self.performance2=sum([i**2 for i in self.member_ability])
        self.wagesum=self.performance
        self.owagesum=self.performance
        
        if loc in core:
            self.nation=[1,0,0,1,0]
        if loc in semi:
            self.nation=[0,1,0,1,0]
        if loc not in semi+core:
            self.nation=[0,0,1,1,0]
        for i in range(1,8):
            for j in range(1,8):
                d=max(abs(i-self.pos[0]),abs(j-self.pos[1]))
                self.gridnumber[d]+=1
        
    
    def recruit(self, model ,n,s,v):
        possible_player=[]
        new_player=[]
        cost=[]
        capability=[]
        for agent in model.p_schedule.agents:
            pl=agent.location
            tl=self.pos
            if pl in self.bondgrid:
                distance=0
            else:
                distance=max(abs(pl[0]-tl[0]),abs(pl[1]-tl[1]))
            # global vision (no restriction of where players from )
            if v==0 and type(agent) == Player and agent.empolyed==False and agent.age<=36:# and distance<=1:
                possible_player.append(agent)
            # local vision (hire players from grids which distance<=1)
            elif v==1 and type(agent) == Player and agent.empolyed==False and agent.age<=36 and distance<=1:
                possible_player.append(agent)
    #    random.shuffle(possible_player)
        

        # use google or-tool SCIP algorithm to find the most valuable set of players for team 
        budget=self.salarybudget-sum(self.member_ability)
        if(budget>0):
            for a in possible_player:
                    pl=a.location
                    tl=self.pos
                    if pl in self.bondgrid:
                        distance=0
                    else:
                        distance=max(abs(pl[0]-tl[0]),abs(pl[1]-tl[1]))
                    # team hires players with noise on their ability
                    see=int(abs(math.floor(np.random.normal(a.ability, 3*distance, size=1)[0])))
                    capability.append(see)
                #    if a.nationality in core:
                    cost.append(int(see+s*distance))
                 #   elif a.nationality not in core+semi:
                 #       cost.append(int(see+s*distance-20))
                    #else:
                    #    cost.append(see+s*0.5)
            llst=[i for i in range(len(cost))]
            llst=sorted(llst, key=lambda y: (capability[y],-cost[y]), reverse=True)
            llst=llst[:n]
            new_player=[]
            spend=0
            for k in llst:
                spend+=cost[k]
            if(budget>spend):
                new_player=llst
            else:
                capacities = [budget,n]
                solver = pywrapknapsack_solver.KnapsackSolver(pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_SCIP_MIP_SOLVER,'Multi-dimensional solver')
                cost2=[cost,[1 for i in range(len(cost))]]
                solver.Init(capability, cost2, capacities)
                computed_value = solver.Solve()
                packed_items = []
                maxi=computed_value
                for i in range(len(capability)):
                    if solver.BestSolutionContains(i):
                       packed_items.append(i)
                new_player=packed_items
           #     print(maxi,new_player)
            
            self.wagesum=sum(self.member_ability)
            c=0
            self.distance=[0,0,0,0,0,0]
            self.nordistance=[0,0,0,0,0,0]
            salary=[]
            ability=[]
            for _ in new_player:
                player=possible_player[_]
                pl=player.location
                tl=self.pos
                d=max(abs(pl[0]-tl[0]),abs(pl[1]-tl[1]))
                self.distance[d]+=1
                player.wage=cost[_]
                c+=cost[_]
                self.model.grid.move_agent(player, self.pos)
                player.location=self.pos
                player.empolyed=True
                self.member.append(player)
                self.member_ability.append(player.ability)
                salary.append(cost[_])
                ability.append(player.ability)
            while len(salary)<7:
                salary.append(0)
            while len(ability)<7:
                ability.append(0)
            self.hireability=ability
            self.hirecost=salary
            self.wagesum+=c
            self.nordistance=[self.distance[x]/self.gridnumber[x] if self.gridnumber[x]!=0 else 0 for x in range(6)]
            
            
        else:
            self.distance=[0,0,0,0,0,0]
            self.nordistance=[0,0,0,0,0,0]
                   
    def cal_entropy(self):
        # calculate team's members' entropy
        n=len(self.member)
        player_nationality=[]
        for agent in self.member:
            player_nationality.append(agent.nationality)
        player_nationality=sorted(player_nationality, key=lambda tup:(tup[0], tup[1]))
        nation=[]
        i=0
        for x in range(1,8):
            for y in range(1,8):
                num=0
                for j in range(i,n):
                    if (x,y)==player_nationality[i]:
                        num+=1
                        i+=1
                nation.append(num)
        sort_nation=sorted(nation,reverse=True)[:n]
        
        entropy=0
        for i in range(len(sort_nation)):
            if sort_nation[i]>0:
                entropy+=(sort_nation[i]/n)*np.log(sort_nation[i]/n)
        self.entropy=-entropy
        self.pnation=sort_nation
      
        
    def step(self,s,rank,bg=20,v=0):
        self.salarybudget=int(self.resource+bg*rank+0.5*(self.salarybudget-self.owagesum))
        self.recruit(model,28-len(self.member),s,v)

        self.performance=sum(self.member_ability)
        self.performance2=sum([i**2 for i in self.member_ability])
        # update team's composition
        self.nation=[0,0,0,0,0]
        for i in self.member:
            if i.nationality ==self.pos:
                self.nation[3]+=1
            if i.nationality in self.bondgrid:
                self.nation[4]+=1
            if i.nationality in core:
                self.nation[0]+=1
            elif i.nationality in semi:
                self.nation[1]+=1
            else:
                self.nation[2]+=1
        nn=[]
        for n in self.nation:
            if n==0:
                nn.append(0)
            else:
                nn.append(n/len(self.member))
        self.nation=nn

        
        self.owagesum=self.wagesum
        self.osalarybudget=self.salarybudget
        
        self.cal_entropy()

#increasing
def rank(inp):
    output = [0] * len(inp)
    for i, x in enumerate(sorted(range(len(inp)), key=lambda y: inp[y])):
        output[x] = i
    return output

# greedy algorithm for hiring
def greedy(n, weight, count, values, weights):
    x=0
    y=count
    lst=[i for i in range(n)]
    item=[]
    z=0
    w=sorted(weights)
    lst=sorted(lst, key=lambda y: (values[y],-weights[y]), reverse=True)
   # lst=sorted(lst, key=lambda y: 0 if weights[y]==0 else values[y]*100/weights[y], reverse=True)
    for i in lst:
        if y==0:
            break
        elif x+weights[i]<=weight-sum(w[:(y-1)]):
                item.append(i)
                y-=1
                z+=values[i]
                x+=weights[i]
        else:
            break
    return z, item


'''MODEL'''
class Football(Model):
    def __init__(self, n_player,n_team,r,bond=0):
        super().__init__()
        self.grid = MultiGrid(8, 8, True)
        self.p_schedule = SimultaneousActivation(self)
        self.t_schedule = SimultaneousActivation(self)
        self.mobile20h= [0 for i in range(49)]
        self.mobile20l= [0 for i in range(49)]
        self.mobile28h= [0 for i in range(49)]
        self.mobile28l= [0 for i in range(49)]
        for i in range(n_player):
            new_player=Player(i, self, state=False)
            self.grid.place_agent(new_player, new_player.location)
            self.p_schedule.add(new_player)
        _id=0
        d=0
        for x in range(2,7):
            for y in range(2,7):
                p=(x,y)
                
                if x==y and x==4:
                    b=[]
                    for i in range(bond):
                        grid=(randint(1,7),randint(1,7))
                        while grid in core+b:
                            grid=(randint(1,7),randint(1,7))
                        b.append(grid)
                    for i in range(n_team):
                        team=Team(_id,self,p,1,r,b)
                        self.t_schedule.add(team)
                        team.init_recruit(self, 28, p)
                        _id+=1
                    
                elif x>2 and x<6 and y>2 and y<6:
                    b=[]
                    if d<12:
                        for i in range(bond):
                            grid=(randint(1,7),randint(1,7))
                            while grid in core+b:
                                grid=(randint(1,7),randint(1,7))
                            b.append(grid)
                    for i in range(n_team):
                        team=Team(_id,self, p,2,r,b)
                        self.t_schedule.add(team)
                        team.init_recruit(self, 28, p)
                        _id+=1
                        d+=1
                    
                else:
                    for i in range(n_team):
                        team=Team(_id,self, p,3,r)
                        self.t_schedule.add(team)
                        team.init_recruit(self, 28, p)
                        _id+=1
        player_list20=[]
        player_list28=[]
        for pl in self.p_schedule.agents:
            if pl.age==20 or pl.age==21:
                player_list20.append([pl,pl.ability])
            elif pl.age==28 or pl.age==29:
                player_list28.append([pl,pl.ability])
        player_list20=sorted(player_list20, key=lambda y: y[1], reverse=True)
        player_list28=sorted(player_list28, key=lambda y: y[1], reverse=True)
       # print(len(player_list))
        xx=0
        a1=[]
        a2=[]
        for pla in player_list20:
            nn=len(player_list20)
            if xx<nn/4:
                pla[0].rank=1
                (m,n)=pla[0].location
                self.mobile20h[(m-1)*7+n-1]+=1
                a1.append(pla[0].ability)
            elif  xx>nn-(nn/4):
                pla[0].rank=2
                (m,n)=pla[0].location
                self.mobile20l[(m-1)*7+n-1]+=1
                a2.append(pla[0].ability)
            xx+=1
        xx=0
        a3=[]
        a4=[]
        for pla in player_list28:
            nn=len(player_list28)
            if xx<nn/4:
                pla[0].rank=3
                (m,n)=pla[0].location
                self.mobile28h[(m-1)*7+n-1]+=1
                a3.append(pla[0].ability)
            elif  xx>nn-(nn/4):
                pla[0].rank=4
                (m,n)=pla[0].location
                self.mobile28l[(m-1)*7+n-1]+=1
                a4.append(pla[0].ability)
            xx+=1
        self.mobile20h.append(sum(a1)/len(a1))
        self.mobile20l.append(sum(a2)/len(a2))
        self.mobile28h.append(sum(a3)/len(a3))
        self.mobile28l.append(sum(a4)/len(a4))

    def generateplayer(self, year,sn,n):
        # adding young player each year
        for i in range(n):
            new_player=Player(sn+n*(year-1)+i+1, self,state=0,r=0)
            self.grid.place_agent(new_player, new_player.location)
            self.p_schedule.add(new_player)

    def cal_mobile(self):
        # record how labeled players' move between grids 
        m20h=[0 for i in range(49)]
        m20l=[0 for i in range(49)]
        m28h=[0 for i in range(49)]
        m28l=[0 for i in range(49)]
        a1=0
        a2=0
        a3=0
        a4=0
        nn=[0,0,0,0]
        for pl in model.p_schedule.agents:
            if pl.rank==1:
                (m,n)=pl.location
                m20h[(m-1)*7+n-1]+=1
                a1+=pl.ability
                nn[0]+=1
            elif pl.rank==2:
                (m,n)=pl.location
                m20l[(m-1)*7+n-1]+=1
                a2+=pl.ability
                nn[1]+=1
            elif pl.rank==3:
                (m,n)=pl.location
                m28h[(m-1)*7+n-1]+=1
                a3+=pl.ability
                nn[2]+=1
            elif pl.rank==4:
                (m,n)=pl.location
                m28l[(m-1)*7+n-1]+=1
                a4+=pl.ability
                nn[3]+=1
        m20h.append(a1/nn[0])
        self.mobile20h=m20h
        m20l.append(a2/nn[1])
        self.mobile20l=m20l
        m28h.append(a3/nn[2])
        self.mobile28h=m28h
        m28l.append(a4/nn[3])
        self.mobile28l=m28l
        
    def step(self, year, s, tp, n_add, bg, vs=0):
        self.p_schedule.step()
        self.generateplayer(year,tp,n_add)
        team_list=[]
        for team in model.t_schedule.agents:
            team_list.append([team,team.performance])
        team_list=sorted(team_list, key=lambda y: y[1], reverse=True)
        
        for i in team_list:
            i[0].fire()
      #      i[0].recruit(model,3)
       # random.shuffle(team_list)
        r=75
        for i in team_list:
    #        print(r)
            i[0].step(s,r,bg,vs)
            r-=1
        self.cal_mobile()
            
        


'''Run'''
data=[]
data_d=[]
data_m=[]
start = timeit.default_timer()
n=21000
r=2000
core=[(3,3),(4,3),(5,3),(3,4),(4,4),(5,4),(3,5),(4,5),(5,5)]
semi=[(2,2),(3,2),(4,2),(5,2),(6,2),(2,3),(2,4),(2,5),(2,6),(3,6),(4,6),(5,6),(6,6),(6,5),(6,4),(6,3)]
bc=20
sc_vs=[[150,1]]#,[60,0]]
bondnumber=[0]#[1,2,3,4,5]

for z in bondnumber:
        for s in sc_vs:
            print(s)
            print(s[0],s[1])
            for t in range(1):
                model=Football(n,3,r,z)
                if t%6==0:
                    print('t='+str(t))
                for i in range(25*3):
                        team=model.t_schedule.agents[i]
                        row_data=[z,s[0],s[1],t,0,team.pos,team.bondgrid, team.resource,team.performance,team.performance2,team.osalarybudget,team.wagesum,team.entropy]#+team.hireability+team.hirecost
                        data.append(row_data)
                        row=[z,s[0],s[1],t,0,team.pos, team.resource,len(team.member)] + team.distance + team.nordistance + team.nation
                mrowh=[z,s[0],s[1],t,0,1]+model.mobile20h
                mrowl=[z,s[0],s[1],t,0,2]+model.mobile20l
                mrowh2=[z,s[0],s[1],t,0,3]+model.mobile28h
                mrowl2=[z,s[0],s[1],t,0,4]+model.mobile28l
                data_m.append(mrowh)
                data_m.append(mrowl)
                data_m.append(mrowh2)
                data_m.append(mrowl2)
             #   print(mrowh2)
                df=DataFrame(data)
                for x in range(30):
                    #print(x)
                    model.step(x+1,s[0],n,int(n/16), bc,s[1])
                    for i in range(25*3):
                            team=model.t_schedule.agents[i]
                            row_data=[z,s[0],s[1],t,x+1,team.pos, team.bondgrid, team.resource,team.performance,team.performance2,team.osalarybudget,team.wagesum,team.entropy]#+team.hireability+team.hirecost
                            data.append(row_data)
                            row=[z,s[0],s[1],t,x+1,team.pos, team.resource,len(team.member)] + team.distance + team.nordistance + team.nation
                            data_d.append(row)
                    data_m.append([z,s[0],s[1],t,x+1,1]+model.mobile20h)
                 #   print([z,s[0],s[1],t,x+1,3]+model.mobile28h)
                    data_m.append([z,s[0],s[1],t,x+1,2]+model.mobile20l)
                    data_m.append([z,s[0],s[1],t,x+1,3]+model.mobile28h)
                    data_m.append([z,s[0],s[1],t,x+1,4]+model.mobile28l)

df=DataFrame(data)
df2=DataFrame(data_d)
df3=DataFrame(data_m)

stop = timeit.default_timer()




print('Time: ', (stop - start)/60)  

#df.to_csv("output_nor.csv")
#df2.to_csv("outputd_nor.csv")
df3.to_csv("outputm_nor.csv")#