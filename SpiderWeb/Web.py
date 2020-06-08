##
## Title:   Web
## Author:  Rubén Ibarrondo (rubenibarrondo@gmail.com)
## Description:
##          Defines the main features of the class Web
##      and its parent class Holder.

import numpy as np
import matplotlib.pyplot as plt

class Holder():
    
    def __init__ (self):
        self.holdlines = []
        
    def __len__(self):
        return len(self.holdlines)

    def __getitem__(self, index):
        return self.holdlines[index]
    
    def __getiter__(self):
        return self.holdlines
    
    def add_holdline(self, orig, dest):
        self.holdlines.append((orig, dest))
        
    def check_holdable(self, point):
        point = np.array(point)
        for holdline in self:
            orig, dest = holdline
            if sum(orig-point)*sum(dest-point) == 0:
                return True
        return False
    
    def plot(self, fig = None, ax = None):
        if not fig:
            for holdline in self:
                x = [holdline[0][0], holdline[1][0]]
                y = [holdline[0][1], holdline[1][1]]
                plt.plot(x,y, "tab:orange")
        else:
            for holdline in self:
                x = [holdline[0][0], holdline[1][0]]
                y = [holdline[0][1], holdline[1][1]]
                ax.plot(x,y, "tab:orange")
        
class Web():
    def __init__(self, holder, genome = []):
        self.wall = holder
        self.web_points = []
        self.DNA = []
        
        if genome:
            for gene in genome:
                self.new_gene(*gene)
                
    def __len__(self):
        """ Number of threads in the web"""
        return len(self.web_points)
    
    def __getitem__(self, index): ## include the wall!
        return self.web_points[index]

    def __getiter__(self): ## include the wall
        return self.web_points

    def __lt__(self, other):
        return True

    def get_genome(self):
        return self.DNA[:]
    
    def crossover(self, other):
        assert len(self.DNA) == len(other.DNA), "self and other must have same length"
        assert self.wall == other.wall, "self and other must hava same wall"
        
        genome1 = self.get_genome()
        genome2 = other.get_genome()

        i = np.random.randint(0, len(genome1))
        new_genome = genome1[:i] + genome2[i:]
        return Web(self.wall, new_genome)
        
    def flyhaunt(self, fly_radius, fly_num, ax = None):
        n = 0
        for x,y in np.random.random((fly_num, 2)):
            if self.checkhaunt([x,y], fly_radius):
                n += 1
                if ax:
                    ax.scatter(x,y, c="tab:green")
            else:
                if ax:
                    ax.scatter(x,y, c="tab:red")
        return n

    def sweep_flyhaunt(self, rarr, fly_num, itnum):
        narr = []
        
        for r in rarr:
            n = 0
            for j in range(itnum):
                n += wb1.flyhaunt(r,fly_num)/fly_num
            n /= itnum
            narr.append(n)
        return np.array(narr)
        
    def checkhaunt(self, point, radius): ## not working
        point = np.array(point)
        radius2 = radius*radius
        d = lambda x,a,b: sum((x-a)*(x-a)) * (1-sum((x-a)*(b-a))**2/sum((x-a)*(x-a))/sum((b-a)*(b-a)))
        for thread in self:
            orig, dest = thread
            if d(point, orig, dest)<radius2:
                if ((point < orig) == (point > dest)).any():
                    return True
        return False

    def mutation(self):
        i, j  = np.random.choice(range(len(self.DNA)), 2, replace=False)
        aux_gene = self.DNA[i]
        self.DNA[i] = self.DNA[j]
        self.DNA[j] = aux_gene
        return self

    def ext_mutation(self):
        genome = self.get_genome()
        nw = Web(self.wall, genome)
        return nw.mutation()
    
    def new_gene(self, int1, prp1, int2, prp2):
        NN = (len(self.web_points)+len(self.wall))
        int1 = int1 % NN
        int2 = int2 % NN

        if int1 < len(self.web_points):
            orig0, dest0 = self.web_points[int1]
        else:
            orig0, dest0 = self.wall[int1-len(self.web_points)]
            
        if int2 < len(self.web_points):
            orig1, dest1 = self.web_points[int2]
        else:
            orig1, dest1 = self.wall[int2-len(self.web_points)]

        neworig = orig0*(1-prp1) + dest0*prp1
        newdest = orig1*(1-prp2) + dest1*prp2
       
        self.DNA.append([int1, prp1, int2, prp2])
        self.web_points.append((neworig,newdest))
        return neworig, newdest

    def rand_web(self, thread_num):
        for n in range(thread_num):
            self.new_randshot()
        return self
    
    def new_randshot(self):
        randind = np.random.choice(range(len(self)+len(self.wall)), 2, replace=False)
        randprp = np.random.random(2)
        return self.new_gene(randind[0], randprp[0], randind[1], randprp[1])
    
    def check_holdable(self,point):
        point = np.array(point)
        if self.wall.check_holdable(point):
            return True
        for thread in self:
            orig, dest = thread
            if sum(orig-point)*sum(dest-point) == 0:
                return True
        return False
    
    def plot(self, fig = None, ax = None):
        if not fig:
            for thread in self:
                x = [thread[0][0], thread[1][0]]
                y = [thread[0][1], thread[1][1]]
                plt.plot(x,y, "tab:blue")
        else:
            for thread in self:
                x = [thread[0][0], thread[1][0]]
                y = [thread[0][1], thread[1][1]]
                ax.plot(x,y, "tab:blue")

def create_circular_holder(r = 1, c = (0,0)):
    cc = Holder()
    for i in range(0, 200):
        theta = i * np.pi/100
        cc.add_holdline(np.array([np.cos(theta),np.sin(theta)]),
                          np.array([np.cos(theta+np.pi/100),np.sin(theta+np.pi/100)]))
    return cc

def create_triang_holder(a = (0.5,0), b = (1,1), c = (0,1)):
    cc = Holder()
    cc.add_holdline(np.array(a), np.array(b))
    cc.add_holdline(np.array(a), np.array(c))
    return cc

def create_rect_holder(p1=(0,0), p2=(1,0), h=1):
    cc = Holder()
    cc.add_holdline(np.array(p1), np.array((p1[0],p1[1]+h)))
    cc.add_holdline(np.array(p2), np.array((p2[0],p2[1]+h)))
    return cc

if __name__ == "__main__":
    thread_num = 10
    generation_num = 100
    wall = create_triang_holder()
    fig, ax = plt.subplots(1, 3)

    generation = []
    
    for i in range(generation_num):
        wb = Web(wall)
        wb.rand_web(thread_num)
        generation.append((wb.flyhaunt(0.01, 100)/100,wb))
    generation.sort()

    for n in range(10):
        alpha = generation[-1][1]
        for i in range(len(generation)):
            wb = alpha.crossover(generation[i][1])
##            wb.mutation()
            generation[i] = (wb.flyhaunt(0.01, 100)/100, wb)
        generation.sort()
    
    print("1 : ", generation[-1][0])
    print("2 : ", generation[-2][0])
    print("3 : ", generation[-3][0])
    
    generation[-1][1].plot(fig, ax[0])
    wall.plot(fig, ax[0])
    generation[-2][1].plot(fig, ax[1])
    wall.plot(fig,ax[1])
    generation[-3][1].plot(fig, ax[2])
    wall.plot(fig,ax[2])
    plt.show()
    
    g = np.array(generation)
    plt.plot(g[:, 0])
    plt.show()
