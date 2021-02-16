import socket
import time
import json

from threading import Thread
from socketserver import ThreadingMixIn

from attack.mgrr_pso_attack import MGRR_PSO_ATTACK

from utils.attack_utils import AttackTarget, AttackResult
from utils.global_best_utils import Global_Best_Utils

client_finish_count = 0
client_failed_count = 0

# Multithreaded Python Server: TCP Server Socket Thread Pool


class ClientThread(Thread):
    def __init__(self, ip, port, conn, client_id):
        Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.conn = conn
        self.client_id = client_id

        print("[+] New server socket thread started for client {} -> {}:{}".format(
            self.client_id, self.ip, self.port))

    def run(self):
        global client_finish_count, client_failed_count
        while True:
            data = self.conn.recv(2048)
            if not data:
                print('Connection to client {} closed'.format(self.client_id))
                break

            json_string_result = data.decode("utf-8")
            print("Received data from client {}: {}".format(
                self.client_id, json_string_result))

            attack_result = AttackResult.from_json_string(json_string_result)
            client_finish_count += 1
            if attack_result.success == False:
                client_failed_count += 1
            # TODO: Handle Client Feedback
            # if condition_finish: break

        self.conn.close()


class AttackThread(Thread):
    def __init__(self, client_connections, attack_target, model_server_url=None, global_best_server_url=None, inc_boundary=None, max_boundary=None):
        Thread.__init__(self)
        self.attack_target = attack_target
        self.client_connections = client_connections
        self.client_total = len(self.client_connections)
        self.mgrr_pso_attack = MGRR_PSO_ATTACK(
            self.attack_target.dataset, model_server_url=model_server_url, global_best_server_url=global_best_server_url)
        self.global_best_utils = Global_Best_Utils(global_best_server_url)
        self.inc_boundary = inc_boundary
        self.max_boundary = max_boundary

    def check_dataset_correctly_predicted(self, attack_index):
        # Check if dataset is correctly predicted
        label, predicted_index, score = self.mgrr_pso_attack.dataset_helper.predict(
            self.mgrr_pso_attack.dataset_helper.datasets[attack_index])

        original_index = self.mgrr_pso_attack.dataset_helper.labels[attack_index]
        correct = predicted_index == original_index
        print('dataset index {} is correctly predicted: {}'.format(
            attack_index, correct))
        return correct, original_index

    def publish_attack(self, attack_target):
        global client_finish_count, client_failed_count

        print('===== Attacking index : {} ======'.format(
            attack_target.attack_index, ))
        # Reset the global best server
        self.global_best_utils.reset_global_best()

        # Send command to all connected clients
        print('Sending attack command to all clients')
        for conn in self.client_connections:
            conn.send(
                attack_target.to_json_string().encode('utf-8'))

        # Wait for all client finished
        print('Waiting all client to finished')
        while client_finish_count < self.client_total:
            time.sleep(1)

        if client_failed_count == client_finish_count:
            client_finish_count = 0
            client_failed_count = 0
            return False
        else:
            client_finish_count = 0
            client_failed_count = 0
            return True
        
        

    def run(self):
        attack_finish_count = 0
        attack_index = self.attack_target.attack_index

        initial_sb = self.attack_target.l_inf
        sb = initial_sb

        while attack_finish_count < self.attack_target.attack_count:

            self.attack_target.l_inf = sb

            print('===== Trying to Attacking index : {} | Remaining Attack : {} ======'.format(
                attack_index, (self.attack_target.attack_count - attack_finish_count)))

            # Check if dataset is correctly predicted
            correct, original_index = self.check_dataset_correctly_predicted(
                attack_index)
                
            while not correct:
                attack_index = attack_index + 1
                self.attack_target.update_attack_index(attack_index)
                correct, original_index = self.check_dataset_correctly_predicted(
                    attack_index)


            if self.attack_target.attack_type == 'untargeted':
                success = self.publish_attack(self.attack_target)

                if not success and not self.inc_boundary == None and not self.max_boundary == None and sb + self.inc_boundary <= self.max_boundary:
                    sb += self.inc_boundary
                    print('Retrying with SB: {}'.format(sb))
                else:
                    attack_finish_count += 1
                    attack_index += 1
                    sb = initial_sb
                    self.attack_target.update_attack_index(attack_index)
                
                time.sleep(1)

            elif self.attack_target.attack_type == 'targeted':
                target_index = 0
                
                # Targeted Attack for all other class
                while target_index < 10:
                    if original_index == target_index:
                        target_index += 1
                        continue
                    
                    self.attack_target.update_target_index(target_index)

                    success = self.publish_attack(self.attack_target)

                    target_index += 1
                    self.attack_target.update_target_index(target_index)
                
                attack_index += 1
                self.attack_target.update_attack_index(attack_index)
                attack_finish_count += 1
            else:
                break


class AttackServer:
    """
    Distributed Attack Server
    """

    def __init__(self, tcp_ip='0.0.0.0', tcp_port=2004, buffer_size=2048, model_server_url=None, global_best_server_url=None, inc_boundary=None, max_boundary=None):
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.buffer_size = buffer_size

        self.model_server_url = model_server_url
        self.global_best_server_url = global_best_server_url

        self.inc_boundary = inc_boundary
        self.max_boundary = max_boundary

        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server.bind((self.tcp_ip, self.tcp_port))

        self.threads = []
        self.conns = []

    def run(self, client_total, attack_target):
        client_count = 0
        while True:
            self.tcp_server.listen(4)
            print("Multithread server: Waiting for connections from clients")
            (conn, (ip, port)) = self.tcp_server.accept()
            new_client_thread = ClientThread(
                ip, port, conn, client_count)
            new_client_thread.start()
            self.threads.append(new_client_thread)
            self.conns.append(conn)
            client_count += 1
            print('client count: {}'.format(client_count))

            if client_count == client_total:
                attack_thread = AttackThread(
                    self.conns, attack_target, self.model_server_url, self.global_best_server_url, self.inc_boundary, self.max_boundary)
                attack_thread.run()
                self.threads.append(attack_thread)
                break

        for t in self.threads:
            t.join()
