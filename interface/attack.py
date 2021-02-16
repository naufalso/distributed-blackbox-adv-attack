from attack.mgrr_pso_attack import MGRR_PSO_ATTACK
from PyInquirer import prompt, Validator, ValidationError, Separator

import matplotlib.pyplot as plt
import regex


class InterfaceAttack:
    def __init__(self):
        pass

    def open_interface(self):
        questions = [
            {
                'type': 'list',
                'name': 'dataset',
                'message': 'Which dataset ?',
                'choices': [
                    'mnist',
                    'mnist-cw',
                    Separator(),
                    'cifar10',
                    'cifar10-cw'
                ]
            },
            {
                'type': 'list',
                'name': 'attack_type',
                'message': 'Targeted or untargeted attack?',
                'choices': ['untargeted', 'targeted']
            },
            {
                'type': 'input',
                'name': 'particle_size',
                'message': 'How many particle size (int)?',
                'validate': IsIntegerValidator
            },
            {
                'type': 'input',
                'name': 'l_inf',
                'message': 'Maximum search boundary (L-Inf) [0.0 ~ 1.0]?',
                'validate': IsFloatValidator
            },
            {
                'type': 'input',
                'name': 'test_index',
                'message': 'Text index (int)? ',
                'validate': IsIntegerValidator
            },
            {
                'type': 'input',
                'name': 'target_label',
                'message': 'Target label index (int)? ',
                'validate': IsIntegerValidator,
                'when': lambda answers: answers['attack_type'] == 'targeted'
            }
        ]

        answers = {'target_label': None}
        answers = prompt(questions, answers=answers)

        self.attack(answers['dataset'],
                    answers['attack_type'], answers['particle_size'], answers['l_inf'], answers['test_index'], answers['target_label'])

    def attack(self, dataset, attack_type, particle_size, l_inf, test_index, target_label=None):
        mgrr_pso_attack = MGRR_PSO_ATTACK(
            dataset, particle_size=int(particle_size))
        if attack_type == 'untargeted':
            err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred, score = mgrr_pso_attack.untargeted_attack(
                int(test_index), float(l_inf), max_iteration=1000, auto_stop=True)

        elif attack_type == 'targeted':
            assert target_label != None
            err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred = mgrr_pso_attack.targeted_attack(
                int(test_index), int(target_label), float(l_inf), max_iteration=1000, auto_stop=True)

        if 'mnist' in dataset:
            label, pred, score = mgrr_pso_attack.dataset_helper.predict(
                pos_best.reshape((28, 28)), with_softmax=mgrr_pso_attack.with_softmax)
            plt.title(label)
            plt.imshow(pos_best.reshape((28, 28)), cmap='gist_gray')
        else:
            label, pred, score = mgrr_pso_attack.dataset_helper.predict(
                pos_best.reshape(mgrr_pso_attack.dataset_helper.shape), with_softmax=mgrr_pso_attack.with_softmax
            )
            plt.title(label)
            plt.imshow(pos_best.reshape(mgrr_pso_attack.dataset_helper.shape))
        plt.show()


class IsIntegerValidator(Validator):
    def validate(self, document):
        ok = regex.match(
            '(?<=\s|^)\d+(?=\s|$)', document.text)
        if not ok:
            raise ValidationError(
                message='Please enter a valid integer',
                cursor_position=len(document.text))  # Move cursor to end


class IsFloatValidator(Validator):
    def validate(self, document):
        ok = regex.match(
            '[+-]?([0-9]*[.])?[0-9]+', document.text)
        if not ok:
            raise ValidationError(
                message='Please enter a valid float',
                cursor_position=len(document.text))  # Move cursor to end
