import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import os
from helpers.google import Google_Vision_Helpers
from attack.mgrr_pso_attack import MGRR_PSO_ATTACK


class Google_Attack(MGRR_PSO_ATTACK):
    # TODO Test this class
    def __init__(self, image_path, particle_size=4, w=0.75, c1=1.0, c2=2.0, md_const=100):
        self.google_helpers = Google_Vision_Helpers()
        self.original_input = self.google_helpers.load_img(image_path)

        MGRR_PSO_ATTACK.__init__(self, 'google', self.original_input.flatten(
        ).shape, particle_size, w, c1, c2, md_const)

    def untargeted_attack(self, original_image_label, L_INF, max_iteration=10000, auto_stop=False):
        original_label = original_image_label

        bounds = (-L_INF, L_INF)

        # Untargeted Attack Single Query
        def evaluate_img(img_flat):
            early_stop = False
            img = img_flat.reshape(self.original_input.shape)
            predictions = self.google_helpers.query_img(img)
            original_label_prediction, is_highest = self.google_helpers.get_score(
                predictions, original_image_label)
            if not is_highest:
                # print('original_label = {} | score = {}'.format(original_label, original_label_prediction))
                early_stop = img_flat
            print('cost : ', original_label_prediction)
            return original_label_prediction, early_stop

        start_time = time.time()
        err_best, pos_best, cost_history, iteration = self.run(self.original_input.flatten(
        ), evaluate_img, bounds, iteration=max_iteration, report=2, early_stop=0.0, auto_early_stop=auto_stop)
        elapsed_time = time.time() - start_time
        print('elapsed time %f s' % (elapsed_time))
        print(err_best, iteration)

        img = pos_best.reshape(self.original_input.shape)
        predictions = self.google_helpers.query_img(img)
        original_label_prediction, is_highest = self.google_helpers.get_score(
            predictions, original_image_label)

        plt.title('{} ({})'.format(
            original_image_label, original_label_prediction))
        plt.imshow(img)
        plt.imsave(self.untargeted_attack_output_dir +
                   original_image_label + '.jpeg', img)
        plt.savefig(self.untargeted_attack_output_dir +
                    original_image_label + '_fig.jpeg')

        distances = self.google_helpers.calculate_distance(
            self.original_input.flatten(), pos_best)
        print(distances)

        return err_best, pos_best, cost_history, iteration, elapsed_time, distances

    def targeted_attack(self, target_label, L_INF, particle_size=4, max_iteration=10000, auto_stop=False):
        bounds = (-L_INF, L_INF)

        # Targeted Attack Single Query
        def evaluate_img(img_flat):
            early_stop = False
            img = img_flat.reshape(self.original_input.shape)
            predictions = self.google_helpers.query_img(img)
            target_prediction, is_highest = self.google_helpers.get_score(
                predictions, target_label)
            cost = 1.0 - target_prediction

            if is_highest:
                print('target_label = {} | score = {}'.format(target_label, cost))
                early_stop = img_flat
            return cost, early_stop

        start_time = time.time()
        err_best, pos_best, cost_history, iteration = self.run(self.original_input.flatten(
        ), evaluate_img, bounds, iteration=max_iteration, report=2, early_stop=0.0, auto_early_stop=auto_stop)
        elapsed_time = time.time() - start_time
        print('elapsed time %f s' % (elapsed_time))
        print(err_best, iteration)

        img = pos_best.reshape(self.original_input.shape)
        predictions = self.google_helpers.query_img(img)
        target_label_prediction, is_highest = self.google_helpers.get_score(
            predictions, target_label)

        distances = self.google_helpers.calculate_distance(
            self.original_input.flatten(), pos_best)
        plt.title('{} ({})'.format(target_label, target_label_prediction))
        plt.imshow(img)
        plt.imsave(self.targeted_attack_output_dir +
                   target_label + '.jpeg', img)
        plt.savefig(self.targeted_attack_output_dir +
                    target_label + '_fig.jpeg')
        print(distances)

        return err_best, pos_best, cost_history, iteration, elapsed_time, distances

    def untargeted_attack_multiple(self, original_image_labels, L_INF, particle_size=4, max_iteration=10000, auto_stop=False):
        original_labels = original_image_labels

        bounds = (-L_INF, L_INF)

        # Untargeted Attack Single Query
        def evaluate_img(img_flat):
            early_stop = False
            img = img_flat.reshape(self.original_input.shape)
            predictions = self.google_helpers.query_img(img)
            total_prediction = self.google_helpers.get_total_score(
                predictions, original_labels)
            if total_prediction == 0:
                print('original_label = {} | score = {}'.format(
                    original_labels, total_prediction))
                early_stop = img_flat
            print('cost : ', total_prediction)
            return total_prediction, early_stop

        start_time = time.time()
        err_best, pos_best, cost_history, iteration = self.run(self.original_input.flatten(
        ), evaluate_img, bounds, iteration=max_iteration, report=10, early_stop=0.0, auto_early_stop=auto_stop)
        elapsed_time = time.time() - start_time
        print('elapsed time %f s' % (elapsed_time))
        print(err_best, iteration)

        img = pos_best.reshape(self.original_input.shape)
        predictions = self.google_helpers.query_img(img)
        total_prediction = self.google_helpers.get_total_score(
            predictions, original_labels)
        plt.title('{} ({})'.format(
            '-'.join(original_labels), total_prediction))
        plt.imshow(img)
        plt.imsave(self.untargeted_attack_output_dir +
                   '-'.join(original_labels) + '_multi.png', img)
        plt.savefig(self.untargeted_attack_output_dir +
                    '-'.join(original_labels) + '_multi_fig.png')

        distances = self.google_helpers.calculate_distance(
            self.original_input.flatten(), pos_best)
        print(distances)

        return err_best, pos_best, cost_history, iteration, elapsed_time, distances

    def untargeted_clean(self, poisoned_image_path, original_image_label, L_INF, particle_size=4, max_iteration=100, auto_stop=False):
        # TODO Complete This codes
        original_label = original_image_label

        poisoned_input = self.google_helpers.load_img(poisoned_image_path)
        poisoned_input_shape = poisoned_input.shape
        poisoned_input_flatten = poisoned_input.flatten()

        bounds = (-L_INF, L_INF)

        # Untargeted Attack Single Query
        def evaluate_img(img_flat):
            early_stop = False
            img = img_flat.reshape(self.original_input.shape)
            predictions = self.google_helpers.query_img(img)
            original_label_prediction, is_highest = self.google_helpers.get_score(
                predictions, original_image_label)
            loss = 100000.0
            if original_label_prediction < 0.3:
                print('original_label = {} | score = {}'.format(
                    original_label, original_label_prediction))
                distances = self.google_helpers.calculate_distance(
                    self.original_input.flatten(), img_flat)
                loss = distances[1]
            # print('cost : ', original_label_prediction)
            return loss, early_stop

        start_time = time.time()
        err_best, pos_best, cost_history, iteration = self.run_with_boundary(self.original_input.flatten(
        ), poisoned_input_flatten, evaluate_img, iteration=max_iteration, report=1, early_stop=0.0, auto_early_stop=auto_stop)
        elapsed_time = time.time() - start_time
        print('elapsed time %f s' % (elapsed_time))
        print(err_best, iteration)

        img = pos_best.reshape(self.original_input.shape)
        predictions = self.google_helpers.query_img(img)
        original_label_prediction, is_highest = self.google_helpers.get_score(
            predictions, original_image_label)

        plt.title('{} ({})'.format(
            original_image_label, original_label_prediction))
        plt.imshow(img)
        plt.imsave(self.untargeted_attack_output_dir +
                   original_image_label + '_clean.jpeg', img)
        plt.savefig(self.untargeted_attack_output_dir +
                    original_image_label + '_clean_fig.jpeg')

        distances = self.google_helpers.calculate_distance(
            self.original_input.flatten(), pos_best)
        print(distances)

        return err_best, pos_best, cost_history, iteration, elapsed_time, distances
