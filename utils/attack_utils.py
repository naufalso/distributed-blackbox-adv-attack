import json
import jsonpickle

from typing import Type

class AttackTarget(object):
    def __init__(self, dataset, attack_type, particle_size, l_inf, max_iteration, attack_index=0, attack_count=1000, target_index=None, *args, **kwargs):
        self.dataset = str(dataset)
        self.attack_type = str(attack_type)
        self.particle_size = int(particle_size)
        self.l_inf = float(l_inf)
        self.max_iteration = int(max_iteration)
        self.attack_count = int(attack_count)
        self.attack_index = int(attack_index)
        if not target_index == None:
            self.target_index = target_index

    def update_attack_index(self, attack_index):
        self.attack_index = int(attack_index)

    def update_target_index(self, target_index):
        self.target_index = target_index

    def to_json_string(self):
        # json_data = {
        #     'dataset': self.dataset,
        #     'attack_type': self.attack_type,
        #     'particle_size': self.particle_size,
        #     'l_inf': self.l_inf,
        #     'max_iteration': self.max_iteration,
        #     'attack_index': self.attack_index
        # }
        return json.dumps(self, default=lambda o: o.__dict__)

    @staticmethod
    def from_json_string(json_string) -> 'AttackTarget':
        # return jsonpickle.decode(json_string)
        # json_data = json.loads(json_string)
        # attack_target = AttackTarget(
        #     json_data['dataset'],
        #     json_data['attack_type'],
        #     json_data['particle_size'],
        #     json_data['l_inf'],
        #     json_data['max_iteration'],
        #     json_data['attack_index']
        # )
        # return attack_target
        json_data =  json.loads(json_string)
        return AttackTarget(**json_data)


class AttackResult:

    def __init__(self, dataset, attack_type, attack_index, original_label, predicted_label, success, probability, iteration, elapsed_time, err_best, l0, l2, l_inf, target_index=None):
        self.dataset = str(dataset)
        self.attack_type = str(attack_type)
        self.attack_index = int(attack_index)
        self.original_label = int(original_label)
        self.predicted_label = int(predicted_label)
        self.success = bool(success)
        self.probability = float(probability)
        self.iteration = int(iteration)
        self.elapsed_time = elapsed_time
        self.err_best = float(err_best)
        self.l0 = float(l0)
        self.l2 = float(l2)
        self.l_inf = float(l_inf)
        if not target_index == None:
            self.target_index = int(target_index)

    def to_json_string(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    @staticmethod
    def from_json_string(json_string):
        json_data = json.loads(json_string)
        return AttackResult(**json_data)
