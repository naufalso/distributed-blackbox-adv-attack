import matplotlib.pyplot as plt
import numpy as np
import time
import os

from pso.distributed_multigroup_pso import Distributed_MGRR_PSO
from helpers.cifar10 import Cifar10_Helpers
from helpers.mnist import Mnist_Helpers
from configure.server_configuration import MODEL_SERVER_URL


class MGRR_PSO_ATTACK(Distributed_MGRR_PSO):

    def __init__(self, dataset_name, flattened_shape=None, particle_size=4, w=0.75, c1=1.0, c2=2.0, md_const=25.0, model_server_url=None, global_best_server_url=None):

        self.untargeted_attack_output_dir = 'result/imgs/{}/untargeted-attack/'.format(
            dataset_name)
        self.targeted_attack_output_dir = 'result/imgs/{}/targeted-attack/'.format(
            dataset_name)
        self.create_output_dir(self.untargeted_attack_output_dir)
        self.create_output_dir(self.targeted_attack_output_dir)

        assert dataset_name == 'mnist' or dataset_name == 'mnist-cw' or dataset_name == 'cifar10' or dataset_name == 'cifar10-cw' or dataset_name == 'google'
        self.dataset_name = dataset_name

        if model_server_url == None:
            model_server_url = MODEL_SERVER_URL

        print('flattened_shape', flattened_shape)


        if not flattened_shape == None:
            Distributed_MGRR_PSO.__init__(
                self, flattened_shape, particle_size=particle_size, w=w, c1=c1, c2=c2, md_const=md_const, global_best_server_url=global_best_server_url)
        else:
            self.with_softmax = False
            if dataset_name == 'cifar10':
                self.dataset_helper = Cifar10_Helpers(
                    model_server_url=model_server_url+'/cifar-10/predict')
            elif dataset_name == 'mnist':
                print('dataset is mnist')
                self.dataset_helper = Mnist_Helpers(
                    model_server_url=model_server_url+'/mnist/predict')
            elif dataset_name == 'cifar10-cw':
                self.dataset_helper = Cifar10_Helpers(
                    model_server_url=model_server_url+'/cifar-10-cw/predict')
                self.with_softmax = True
            elif dataset_name == 'mnist-cw':
                self.dataset_helper = Mnist_Helpers(
                    model_server_url=model_server_url+'/mnist-cw/predict')
                self.with_softmax = True

            Distributed_MGRR_PSO.__init__(
                self, self.dataset_helper.flattened_shape, particle_size=particle_size, w=w, c1=c1, c2=c2, md_const=md_const, global_best_server_url=global_best_server_url)

        print('MGRR_PSO ATTACK for {} is loaded'.format(self.dataset_name))

    def create_output_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def untargeted_attack(self, test_image_index, L_INF, max_iteration=10000, auto_stop=False):
        original_label = np.asscalar(
            self.dataset_helper.labels[test_image_index])
        original_input_flatten = self.dataset_helper.datasets[test_image_index].flatten(
        )
        bounds = (-L_INF, L_INF)

        # Untargeted Attack Batch
        def evaluate_imgs(imgs_flat):
            early_stop = False
            imgs = imgs_flat.reshape(self.dataset_helper.get_batch_shape(
                imgs_flat.shape[0]))  # TODO: Fix This
            predictions = self.dataset_helper.query(imgs)

            errs = np.zeros(self.particle_size)
            early_stop_pos = []
            early_stop_errs = []

            for i in range(self.particle_size):
                errs[i] = predictions[i][original_label]
                predicted_label = np.argmax(predictions[i])

                if original_label != predicted_label:
                    early_stop_errs.append(errs[i])
                    early_stop_pos.append(imgs_flat[i])

            if len(early_stop_errs) > 0:
                print('Early Stop Count : {}'.format(len(early_stop_errs)))
                lowest_err_idx = np.argmin(early_stop_errs)
                early_stop = early_stop_pos[lowest_err_idx]

            return errs, early_stop

        print('Original Label', original_label)
        print('Input Shape', original_input_flatten.shape)

        start_time = time.time()
        err_best, pos_best, cost_history, iteration = self.run_batch(
            original_input_flatten, evaluate_imgs, bounds, iteration=max_iteration, report=10, early_stop=0.0, auto_early_stop=auto_stop)
        elapsed_time = time.time() - start_time

        print('elapsed time %f s' % (elapsed_time))
        print(err_best, iteration)

        label, pred, score = self.dataset_helper.predict(pos_best.reshape(
            self.dataset_helper.shape), with_softmax=self.with_softmax)
        distances = self.dataset_helper.calculate_distance(
            self.dataset_helper.datasets[test_image_index].flatten(), pos_best)

        plt.title(label + ' | L1: {:.3f}, L2: {:.3f}, L-Inf: {:.3f}'.format(
            distances[0], distances[1], distances[2]))
        if 'mnist' in self.dataset_name:
            plt.imshow(pos_best.reshape(
                [self.dataset_helper.shape[0], self.dataset_helper.shape[1]]), cmap='gist_gray')
            plt.imsave(self.untargeted_attack_output_dir + str(test_image_index) +17 '({})'.format(L_INF) +
                       '.png', pos_best.reshape((28, 28)), cmap='gist_gray')
        else:
            plt.imshow(pos_best.reshape(self.dataset_helper.shape))
            plt.imsave(self.untargeted_attack_output_dir + str(test_image_index) + '({})'.format(L_INF) +
                       '.png', pos_best.reshape(self.dataset_helper.shape))
        plt.savefig(self.untargeted_attack_output_dir +
                    str(test_image_index) + '_fig.png')

        print(distances)
        return err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred, score

    def targeted_attack(self, test_image_index, targeted_label, L_INF, max_iteration=10000, auto_stop=False):
        original_label = np.asscalar(
            self.dataset_helper.labels[test_image_index])
        original_input_flatten = self.dataset_helper.datasets[test_image_index].flatten(
        )
        bounds = (-L_INF, L_INF)

        # Targeted Attack Batch
        def evaluate_imgs(imgs_flat):
            early_stop = False
            imgs = imgs_flat.reshape(
                self.dataset_helper.get_batch_shape(imgs_flat.shape[0]))
            predictions = self.dataset_helper.query(imgs)

            errs = np.zeros(self.particle_size)
            early_stop_pos = []
            early_stop_errs = []

            for i in range(self.particle_size):
                if self.with_softmax:
                    target_max_prob = 100.0
                else:
                    target_max_prob = 1.0
                errs[i] = target_max_prob - predictions[i][targeted_label]
                predicted_label = np.argmax(predictions[i])

                if predicted_label == targeted_label:
                    early_stop_errs.append(errs[i])
                    early_stop_pos.append(imgs_flat[i])

            if len(early_stop_errs) > 0:
                print('Early Stop Count : {}'.format(len(early_stop_errs)))
                lowest_err_idx = np.argmin(early_stop_errs)
                early_stop = early_stop_pos[lowest_err_idx]

            return errs, early_stop

        start_time = time.time()
        err_best, pos_best, cost_history, iteration = self.run_batch(
            original_input_flatten, evaluate_imgs, bounds, iteration=max_iteration, report=10, early_stop=0.0, auto_early_stop=auto_stop)
        elapsed_time = time.time() - start_time

        print('elapsed time %f s' % (elapsed_time))
        print(err_best, iteration)

        label, pred, score = self.dataset_helper.predict(pos_best.reshape(
            self.dataset_helper.shape), with_softmax=self.with_softmax)
        distances = self.dataset_helper.calculate_distance(
            self.dataset_helper.datasets[test_image_index].flatten(), pos_best)

        plt.title(label + ' | L1: {:.3f}, L2: {:.3f}, L-Inf: {:.3f}'.format(
            distances[0], distances[1], distances[2]))
        if 'mnist' in self.dataset_name:
            plt.imshow(pos_best.reshape(
                [self.dataset_helper.shape[0], self.dataset_helper.shape[1]]), cmap='gist_gray')
            plt.imsave(self.targeted_attack_output_dir + str(test_image_index) + '_' + str(targeted_label) + '({})'.format(L_INF) +
                       '.png', pos_best.reshape((28, 28)), cmap='gist_gray')
        else:
            plt.imshow(pos_best.reshape(self.dataset_helper.shape))
            plt.imsave(self.targeted_attack_output_dir + str(test_image_index) + '_' + str(targeted_label) + '({})'.format(L_INF) +
                       '.png', pos_best.reshape(self.dataset_helper.shape))
        plt.savefig(self.targeted_attack_output_dir +
                    str(test_image_index) + '_fig.png')

        print(distances)
        return err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred

    def untargeted_clean(self, test_image_index, perturbated_image, L_INF, max_iteration=10000, auto_stop=False):
        original_label = np.asscalar(
            self.dataset_helper.label_names[test_image_index])
        original_input_flatten = self.dataset_helper.datasets[test_image_index].flatten(
        )

        bounds = (-L_INF, L_INF)

        # Untargeted Attack Single Query
        def evaluate_img(img_flat):
            early_stop = False
            imgs = img_flat.reshape(self.dataset_helper.single_batch_shape)

            predictions = self.dataset_helper.query(imgs)
            predicted_label = np.argmax(predictions)
            loss = 10000.0

            if original_label != predicted_label:
                distances = self.dataset_helper.calculate_distance(
                    original_input_flatten, img_flat)
                loss = distances[1]

            return loss, early_stop

        start_time = time.time()
        err_best, pos_best, cost_history, iteration = self.run_with_boundary(original_input_flatten, perturbated_image.flatten(
        ), evaluate_img, iteration=max_iteration, report=1, early_stop=0.0, auto_early_stop=auto_stop)
        elapsed_time = time.time() - start_time

        print('elapsed time %f s' % (elapsed_time))
        print(err_best, iteration)

        label, pred, score = self.dataset_helper.predict(pos_best.reshape(
            self.dataset_helper.shape), with_softmax=self.with_softmax)
        distances = self.dataset_helper.calculate_distance(
            self.dataset_helper.datasets[test_image_index].flatten(), pos_best)

        plt.title(label + ' | L1: {:.3f}, L2: {:.3f}, L-Inf: {:.3f}'.format(
            distances[0], distances[1], distances[2]))
        if 'mnist' in self.dataset_name:
            plt.imshow(pos_best.reshape(
                self.dataset_helper.shape), cmap='gist_gray')
            plt.imsave(self.untargeted_attack_output_dir + str(test_image_index) +
                       '_clean.png', pos_best.reshape(self.dataset_helper.shape, cmap='gist_gray'))
        else:
            plt.imshow(pos_best.reshape(self.dataset_helper.shape))
            plt.imsave(self.untargeted_attack_output_dir + str(test_image_index) +
                       '.png', pos_best.reshape(self.dataset_helper.shape))
        plt.savefig(self.untargeted_attack_output_dir +
                    str(test_image_index) + '_clean_fig.png')

        print(distances)

        return err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred
