import numpy as np

class MGRR_PSO:
    def __init__(self, input_shape, bound, costFunc, costFuncBatch, particle_size=8, w=0.75, c1=1.0, c2=2.0, md_const=25.0):
        print('Initialize MGRR PSO')
        np.random.seed(1234)
        self.costFunc = costFunc
        self.costFuncBatch = costFuncBatch
        self.input_shape = input_shape
        self.bound = bound
        self.particle_size = particle_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.md_const = md_const
        self.early_stop = False
        self.init_multi_group()
        
    def init_particle(self, init_position):
        # Particle Variable
        # Shape = (particle_size, input_shape) => 2D
        self.position = np.tile(np.expand_dims(init_position, axis=0), [self.particle_size, 1])
        self.velocity = np.random.uniform(-1.0, 1.0, (self.particle_size, self.input_shape[0]))
        self.pos_best = np.tile(np.expand_dims(init_position, axis=0), [self.particle_size, 1])
        self.lower_bound = np.clip(np.add(np.tile(np.expand_dims(init_position, 0), [self.particle_size, 1]), self.bound[0]), 0.0, 1.0)
        self.upper_bound = np.clip(np.add(np.tile(np.expand_dims(init_position, 0), [self.particle_size, 1]), self.bound[1]), 0.0, 1.0)
        # Shape = (particle_size) => 1D
        self.err_best = np.full((self.particle_size), np.finfo(np.float32).max) # Best Error Individual
        self.err      = np.full((self.particle_size), np.finfo(np.float32).max) # Error Individual
        # Global Variable
        self.global_pos_best = np.tile(np.expand_dims(init_position, 0), [self.particle_size, 1])
        self.global_err_best = np.finfo(np.float32).max

    def init_multi_group(self):
        self.c1_half = np.full([int(self.particle_size/2), 1], self.c1)
        self.c2_half = np.full([int(self.particle_size/2), 1], self.c2)
        self.c1_mg = np.concatenate((self.c1_half, self.c2_half), axis=0)
        self.c2_mg = np.concatenate((self.c2_half, self.c1_half), axis=0)

    def init_random_position(self):
        self.position = np.random.uniform(0, 1, (self.particle_size, self.input_shape[0])) * (self.upper_bound - self.lower_bound) + self.lower_bound

    def update_velocity(self):
        self.vel_cognitive = self.c1_mg * np.random.uniform(0.0, 1.0, (self.particle_size, self.input_shape[0])) * np.subtract(self.pos_best, self.position)
        self.vel_social = self.c2_mg * np.random.uniform(0.0, 1.0, (self.particle_size, self.input_shape[0])) * np.subtract(self.global_pos_best, self.position)
        self.vel_momentum = self.w * self.velocity
        self.velocity = self.vel_momentum + self.vel_cognitive + self.vel_social

    def update_position(self):
        self.position = np.clip(np.add(self.position, self.velocity), self.lower_bound, self.upper_bound)

    def evaluate_cost(self):
        for i in range(self.particle_size):
            particle_cost, early_stop = self.costFunc(self.position[i])
            if early_stop:
                self.early_stop = True
            self.err[i] = particle_cost
            if particle_cost < self.err_best[i]: # Update Local Best
                self.pos_best[i] = self.position[i]
                self.err_best[i] = particle_cost
                
         # Update Global Best
        self.global_err_best = np.amin(self.err_best)
        self.best_global_pos_index = np.argmin(self.err_best)
        self.global_pos_best = np.tile(np.expand_dims(self.position[self.best_global_pos_index], 0), [self.particle_size, 1])

    def evaluate_cost_batch(self):
        self.err, early_stop = self.costFuncBatch(self.position)
        if early_stop:
            self.early_stop = True
        for i in range(self.particle_size):
            if self.err[i] < self.err_best[i]: # Update Local Best
                self.pos_best[i] = self.position[i]
                self.err_best[i] = self.err[i]
                
         # Update Global Best
        self.global_err_best = np.amin(self.err_best)
        self.best_global_pos_index = np.argmin(self.err_best)
        self.global_pos_best = np.tile(np.expand_dims(self.position[self.best_global_pos_index], 0), [self.particle_size, 1])

    def redestribute_particles(self):
        # MGRR-PSO3: Find Average of All Particle Cost
        self.err_mean =np.mean(self.err)
        self.diff = np.abs(self.global_err_best - self.err_mean)
        
        # MGRR-PSO4: Calculate minumum difference tolerance
        self.md = np.abs(self.global_err_best / self.md_const) 

        if self.diff < self.md:
            for i in np.arange(self.particle_size//4, self.particle_size*3//4):
                rd = np.random.uniform(0.0, 1.0, self.input_shape[0])
                for j in range(self.input_shape[0]):
                    if rd[j] >= 0.5:
                        self.position[i][j] = np.random.uniform(0,1) * (self.upper_bound[i][j] - self.lower_bound[i][j]) + self.lower_bound[i][j]
            return True
        return False
                

    def run(self, X, iteration=100, verbose=False, early_stop = 0.0):
        self.cost_history = []
        self.init_particle(X)
        self.evaluate_cost()
        redestribute_count = 0

        for i in range(iteration):
            self.update_velocity()
            self.update_position()
            self.evaluate_cost()
           
            is_destributed = self.redestribute_particles()
            if is_destributed:
                redestribute_count = redestribute_count + 1
            if verbose and i%(iteration/20)==0:
                print('iter %d - redestribute %d: ' % (i, redestribute_count), self.global_err_best)
                redestribute_count = 0
            self.cost_history.append(self.global_err_best)
            if self.global_err_best <= early_stop:
                break

        return self.global_err_best, self.global_pos_best[0], self.cost_history, i

    def run_batch(self, X, iteration=100, verbose=False, early_stop = 0.0, auto_early_stop = False):
        self.cost_history = []
        self.init_particle(X)
        # self.init_random_position()
        self.evaluate_cost_batch()
        redestribute_count = 0

        for i in range(iteration):
            self.update_velocity()
            self.update_position()
            self.evaluate_cost_batch()
            is_destributed = self.redestribute_particles()
            if is_destributed:
                redestribute_count = redestribute_count + 1
            if verbose and i%(iteration/20)==0:
                print('iter %d - redestribute %d: ' % (i, redestribute_count), self.global_err_best)
                redestribute_count = 0
            self.cost_history.append(self.global_err_best)
            if self.global_err_best <= early_stop:
                break
            if auto_early_stop and self.early_stop:
                break

        return self.global_err_best, self.global_pos_best[0], self.cost_history, i