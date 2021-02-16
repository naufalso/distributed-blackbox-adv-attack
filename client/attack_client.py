import socket
import signal
import sys
import json
import os

from csv import DictWriter

from utils.attack_utils import AttackTarget, AttackResult
from attack.mgrr_pso_attack import MGRR_PSO_ATTACK


class AttackClient:
    """
    Distributed Attack Client
    """

    def __init__(self, client_id, tcp_host=socket.gethostname(), tcp_port=2004, buffer_size=2048, model_server_url=None, global_best_server_url=None):

        self.client_id = client_id
        self.host = tcp_host
        self.port = tcp_port
        self.buffer_size = buffer_size

        self.model_server_url = model_server_url
        self.global_best_server_url = global_best_server_url

        # Initialize Socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Handle signal interupt
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(sig, frame):
        # Close socket when the app is forcely closed
        print('You pressed Ctrl+C!')
        self.client_socket.close()
        sys.exit(0)

    def create_output_dir(self, directory):
        # Create output dir if not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

    def write_to_csv(self, attack_result: AttackResult):
        # Append attack result to csv file

        # Declare output file name and dirs
        file_name = '{}_{}_client_{}.csv'.format(
            attack_result.dataset, attack_result.attack_type, self.client_id)
        file_dir = 'result/data/{}/{}/'.format(
            attack_result.dataset, attack_result.attack_type)
        file_loc = file_dir + file_name

        self.create_output_dir(file_dir)

        # Convert obj to python dict
        json_data = json.loads(attack_result.to_json_string())

        file_not_exist = not os.path.exists(file_loc)

        # Open file in append mode
        with open(file_loc, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            dict_writer = DictWriter(write_obj, fieldnames=json_data.keys())

            # Append header if not exist
            if file_not_exist:
                dict_writer.writeheader()

            # Add dictionary as row in the csv
            dict_writer.writerow(json_data)

    def run(self):
        # Connect socket to the attack server
        self.client_socket.connect((self.host, self.port))

        # TODO add condition when finished
        while True:

            # Waiting attack instruction from server
            instruction_data = self.client_socket.recv(self.buffer_size)
            instruction_data.decode('utf-8')

            # Convert to attack target json
            attack_target = AttackTarget.from_json_string(instruction_data)

            # Initialize MGRR-PSO Attack
            mgrr_pso_attack = MGRR_PSO_ATTACK(
                attack_target.dataset, particle_size=attack_target.particle_size, model_server_url=self.model_server_url, global_best_server_url=self.global_best_server_url)

            if attack_target.attack_type == 'untargeted':
                err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred, score = mgrr_pso_attack.untargeted_attack(
                    attack_target.attack_index, attack_target.l_inf, max_iteration=attack_target.max_iteration, auto_stop=True)

                original_label = mgrr_pso_attack.dataset_helper.labels[attack_target.attack_index]

                # Combine the outputs as AttackResult Class
                attack_result = AttackResult(
                    attack_target.dataset,
                    attack_target.attack_type,
                    attack_target.attack_index,
                    original_label,
                    pred,
                    original_label != pred,
                    score,
                    iteration,
                    elapsed_time,
                    err_best,
                    distances[0],
                    distances[1],
                    distances[2]
                )

                # Write to CSV
                self.write_to_csv(attack_result)

                # Send the result to Attack Server
                self.client_socket.send(
                    attack_result.to_json_string().encode('utf-8'))

            elif attack_target.attack_type == 'targeted':
                err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred = mgrr_pso_attack.targeted_attack(
                    int(attack_target.attack_index), int(attack_target.target_index), float(attack_target.l_inf), max_iteration=attack_target.max_iteration, auto_stop=True)
                
                original_label = mgrr_pso_attack.dataset_helper.labels[attack_target.attack_index]

                # Combine the outputs as AttackResult Class
                attack_result = AttackResult(
                    attack_target.dataset,
                    attack_target.attack_type,
                    attack_target.attack_index,
                    original_label,
                    pred,
                    original_label != pred,
                    score,
                    iteration,
                    elapsed_time,
                    err_best,
                    distances[0],
                    distances[1],
                    distances[2],
                    attack_target.target_index
                )

                # Write to CSV
                self.write_to_csv(attack_result)

                # Send the result to Attack Server
                self.client_socket.send(
                    attack_result.to_json_string().encode('utf-8'))
                
            del mgrr_pso_attack


            # TODO Continue
