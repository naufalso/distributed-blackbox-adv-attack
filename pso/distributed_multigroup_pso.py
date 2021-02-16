import numpy as np
import json
import requests
from utils.global_best_utils import Global_Best_Utils
from utils.base64_util import Base64_Utils

class Distributed_MGRR_PSO:
    def __init__(self, input_shape, particle_size=4, w=0.75, c1=1.0, c2=2.0, md_const=10.0, global_best_server_url=None):
        print('Initialize MGRR PSO', particle_size, input_shape)
        # np.random.seed(1234)
        self.input_shape = input_shape
        self.particle_size = particle_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.md_const = md_const
        self.early_stop = False
        self.init_multi_group()
        self.global_best_utils = Global_Best_Utils(global_best_server_url)
        self.base64_utils = Base64_Utils()

    def is_numpy(self, obj):
        return type(obj).__module__ == np.__name__
        
    def init_particle(self, init_position, ref_position = False, negative_bound = False, positive_bound = False):
        # Particle Variable
        # Shape = (particle_size, input_shape) => 2D
        print('init_position', init_position.dtype)
        self.position = np.tile(np.expand_dims(init_position, axis=0), [self.particle_size, 1]).astype(np.float32)
        self.velocity = np.random.uniform(-1.0, 1.0, (self.particle_size, self.input_shape[0])).astype(np.float32)
        self.pos_best = np.tile(np.expand_dims(init_position, axis=0), [self.particle_size, 1]).astype(np.float32)

        if not self.is_numpy(negative_bound) and not self.is_numpy(positive_bound) and not self.is_numpy(ref_position):
            print('LINF: ', self.bound)
            self.lower_bound = np.clip(np.add(np.tile(np.expand_dims(init_position, 0), [self.particle_size, 1]), self.bound[0]), 0.0, 1.0).astype(np.float32)
            self.upper_bound = np.clip(np.add(np.tile(np.expand_dims(init_position, 0), [self.particle_size, 1]), self.bound[1]), 0.0, 1.0).astype(np.float32)
        else:
            self.lower_bound = np.clip(np.add(np.tile(np.expand_dims(ref_position, 0), [self.particle_size, 1]), negative_bound), 0.0, 1.0).astype(np.float32)
            self.upper_bound = np.clip(np.add(np.tile(np.expand_dims(ref_position, 0), [self.particle_size, 1]), positive_bound), 0.0, 1.0).astype(np.float32)
        # Shape = (particle_size) => 1D
        self.err_best = np.full((self.particle_size), np.finfo(np.float32).max, dtype=np.float32) # Best Error Individual
        self.err      = np.full((self.particle_size), np.finfo(np.float32).max, dtype=np.float32) # Error Individual
        # Global Variable
        self.global_pos_best = np.tile(np.expand_dims(init_position, 0), [self.particle_size, 1]).astype(np.float32)
        self.global_err_best = np.finfo(np.float32).max

        global_best, global_pos_best = self.query_server_global_best()
        if global_pos_best.shape[0] == 0:
            self.global_pos_best = np.tile(np.expand_dims(init_position, 0), [self.particle_size, 1]).astype(np.float32)
            self.global_err_best = np.finfo(np.float32).max
            self.update_server_global_best(self.global_err_best, self.global_pos_best[0])

    def init_multi_group(self):
        self.c1_half = np.full([int(self.particle_size/2), 1], self.c1, dtype=np.float32)
        self.c2_half = np.full([int(self.particle_size/2), 1], self.c2, dtype=np.float32)
        self.c1_mg = np.concatenate((self.c1_half, self.c2_half), axis=0)
        self.c2_mg = np.concatenate((self.c2_half, self.c1_half), axis=0)

    def init_random_position(self):
        self.position = np.random.uniform(0, 1, (self.particle_size, self.input_shape[0])).astype(np.float32) * (self.upper_bound - self.lower_bound) + self.lower_bound

    def update_velocity(self):
        # print('update_velocity')
        self.vel_cognitive = self.c1_mg * np.random.uniform(0.0, 1.0, (self.particle_size, self.input_shape[0])).astype(np.float32) * np.subtract(self.pos_best, self.position)
        self.vel_social = self.c2_mg * np.random.uniform(0.0, 1.0, (self.particle_size, self.input_shape[0])).astype(np.float32) * np.subtract(self.global_pos_best, self.position)
        self.vel_momentum = self.w * self.velocity
        self.velocity = self.vel_momentum + self.vel_cognitive + self.vel_social

    def update_position(self):
        # print('update_position')
        self.position = np.clip(np.add(self.position, self.velocity), self.lower_bound, self.upper_bound)

    def evaluate_cost(self, start_particle_index = 0, last_particle_index = -1):
        if last_particle_index < 0:
            last_particle_index = self.particle_size
        # print('evaluate_cost')
        global_best, global_best_pos = self.query_server_global_best()
        for i in range(start_particle_index, last_particle_index):
            particle_cost, early_stop = self.costFunc(self.position[i])
            self.err[i] = particle_cost
            if particle_cost < self.err_best[i]: # Update Local Best
                self.pos_best[i] = self.position[i]
                self.err_best[i] = particle_cost
            if self.is_numpy(early_stop):
                self.early_stop = early_stop
                break
                
         # Update Global Best
        if(np.amin(self.err_best) < global_best):
            self.global_err_best = np.amin(self.err_best)
            self.best_global_pos_index = np.argmin(self.err_best)
            self.global_pos_best = np.tile(np.expand_dims(self.position[self.best_global_pos_index], 0), [self.particle_size, 1])
            self.update_server_global_best(self.global_err_best, self.global_pos_best[0])

    def evaluate_cost_batch(self):
        # print('evaluate_cost_batch')
        global_best, global_best_pos = self.query_server_global_best()
        self.err, early_stop = self.costFuncBatch(self.position)
        if self.is_numpy(early_stop):
            self.early_stop = early_stop
        for i in range(self.particle_size):
            if self.err[i] < self.err_best[i]: # Update Local Best
                self.pos_best[i] = self.position[i]
                self.err_best[i] = self.err[i]
                
         # Update Global Best
        if(np.amin(self.err_best) < global_best):
            self.global_err_best = np.amin(self.err_best)
            self.best_global_pos_index = np.argmin(self.err_best)
            self.global_pos_best = np.tile(np.expand_dims(self.position[self.best_global_pos_index], 0), [self.particle_size, 1])
            self.update_server_global_best(self.global_err_best, self.global_pos_best[0])

    def redestribute_particles(self, original = False):
        # print('redestribute_particles')
        # MGRR-PSO3: Find Average of All Particle Cost
        self.err_mean = np.mean(self.err_best)
        self.diff = np.abs(self.global_err_best - self.err_mean)
        
        # MGRR-PSO4: Calculate minumum difference tolerance
        self.md = np.abs(self.global_err_best / self.md_const) 

        if self.diff < self.md: #or self.is_numpy(original):
            for i in np.arange(self.particle_size//4, self.particle_size*3//4):
                res_count = 0
                if self.is_numpy(original):
                    for j in range(self.input_shape[0]):
                        if np.abs(self.position[i][j] - original[j]) > 0.004 and np.random.uniform(0,1) > 0.90:
                            res_count += 1
                            # self.position[i][j] = np.random.uniform(0,1) * (self.upper_bound[i][j] - self.lower_bound[i][j]) + self.lower_bound[i][j]
                            self.position[i][j] = np.random.rand() * (np.maximum(self.position[i][j], original[j]) - np.minimum(self.position[i][j], original[j])) + np.minimum(self.position[i][j], original[j])  # (self.position[i][j] + original[j]) * 0.5
                            # self.position[i][j] = (self.position[i][j] + original[j]) * 0.5
                else:
                    rd = np.random.uniform(0.0, 1.0, self.input_shape[0])
                    for j in range(self.input_shape[0]):
                        if rd[j] <= 0.5 * np.random.uniform(0, 1):
                            self.position[i][j] = np.random.uniform(0,1) * (self.upper_bound[i][j] - self.lower_bound[i][j]) + self.lower_bound[i][j]
                # print(res_count)
                # self.evaluate_cost(start_particle_index=self.particle_size//4, last_particle_index=self.particle_size*3//4)

            return True
        return False

    def query_server_global_best(self):
        global_best, global_pos_best = self.global_best_utils.query_global_best()
        if global_pos_best != []:
            self.global_pos_best = np.tile(np.expand_dims(global_pos_best, 0), [self.particle_size, 1])
            self.global_err_best = global_best
        return global_best, global_pos_best

    def update_server_global_best(self, global_best, global_best_pos):
        return self.global_best_utils.update_global_best(global_best, global_best_pos)
                

    def run(self, X, costFunc, bound, iteration=100, report=20, early_stop = 0.0, auto_early_stop = False, max_auto_terminate = 0):
        print('Auto early stop : ', auto_early_stop)
        self.costFunc = costFunc
        self.bound = bound

        self.cost_history = []
        self.init_particle(X)
        self.init_random_position()
        self.evaluate_cost()
        redestribute_count = 0
        self.last_global_best = self.global_err_best

        for i in range(iteration):
            self.update_velocity()
            self.update_position()
            self.evaluate_cost()
           
            self.cost_history.append(self.global_err_best)

            if auto_early_stop and type(self.early_stop).__module__ == np.__name__:
                print('result shape: ', self.early_stop.shape)
                return self.global_err_best, self.early_stop, self.cost_history, i

            if self.global_err_best <= early_stop:
                break

            if max_auto_terminate > 0 and i % max_auto_terminate == 0:
                if self.last_global_best == self.global_err_best:
                    break
                self.last_global_best = self.global_err_best

            is_destributed = self.redestribute_particles()
            if is_destributed:
                redestribute_count = redestribute_count + 1
            if report > 0 and i%report==0:
                print('iter %d - redestribute %d: ' % (i, redestribute_count), self.global_err_best)
                redestribute_count = 0

        return self.global_err_best, self.global_pos_best[0], self.cost_history, i

    def run_batch(self, X, costFuncBatch, bound, iteration=100, report=20, early_stop = 0.0, auto_early_stop = False, max_auto_terminate = 0):
        self.costFuncBatch = costFuncBatch
        self.bound = bound

        self.cost_history = []
        self.init_particle(X)
        self.init_random_position()
        self.evaluate_cost_batch()
        redestribute_count = 0

        for i in range(iteration):
            self.update_velocity()
            self.update_position()
            self.evaluate_cost_batch()
            
            self.cost_history.append(self.global_err_best)
            if auto_early_stop and type(self.early_stop).__module__ == np.__name__:
                print('Early Stop!')
                return self.global_err_best, self.early_stop, self.cost_history, i

            if self.global_err_best <= early_stop:
                break
          
            if max_auto_terminate > 0 and i % max_auto_terminate == 0:
                if self.last_global_best == self.global_err_best:
                    break
                self.last_global_best = self.global_err_best

            is_destributed = self.redestribute_particles()
            if is_destributed:
                redestribute_count = redestribute_count + 1
            if report > 0 and i%report==0:
                print('iter %d - redestribute %d: ' % (i, redestribute_count), self.global_err_best)
                redestribute_count = 0

        # TODO check why final early stop is not as expected
        return self.global_err_best, self.global_pos_best[0], self.cost_history, i

    def run_with_boundary(self, X, X_ref, costFunc, iteration=100, report=20, early_stop = 0.0, auto_early_stop = False, max_auto_terminate = 0):
        diff_img = X_ref - X
        negative_bound = np.clip(np.copy(diff_img), -1., 0.)
        positive_bound = np.clip(np.copy(diff_img), 0., 1.)
        self.costFunc = costFunc

        self.cost_history = []
        self.init_particle(X_ref, X, negative_bound, positive_bound)
        # self.init_random_position()
        self.evaluate_cost()
        redestribute_count = 0
        self.last_global_best = self.global_err_best

        for i in range(iteration):
            self.update_velocity()
            self.update_position()
            self.evaluate_cost()
           
            is_destributed = self.redestribute_particles(X)
            if is_destributed:
                redestribute_count = redestribute_count + 1
                # self.evaluate_cost()
            if report > 0 and i%report==0:
                print('iter %d - redestribute %d: ' % (i, redestribute_count), self.global_err_best)
                redestribute_count = 0
            self.cost_history.append(self.global_err_best)

            if auto_early_stop and type(self.early_stop).__module__ == np.__name__:
                print('result shape: ', self.early_stop.shape)
                return self.global_err_best, self.early_stop, self.cost_history, i

            if self.global_err_best <= early_stop:
                break

            if max_auto_terminate > 0 and i % max_auto_terminate == 0:
                if self.last_global_best == self.global_err_best:
                    break
                self.last_global_best = self.global_err_best

        return self.global_err_best, self.global_pos_best[0], self.cost_history, i
