# import sys
# import numpy as np
# from utils.global_best_utils import Global_Best_Utils
# from server.ai_model_server import app as model_server
# from attack.mgrr_pso_attack import MGRR_PSO_ATTACK
# from attack.google_attack import Google_Attack
# from utils.attack_utils import AttackTarget

# from utils.base64_util import Base64_Utils
# from helpers.mnist import Mnist_Helpers
# # from interface.attack import InterfaceAttack
# # from interface.server import InterfaceServer

# from server.attack_server import AttackServer
# from client.attack_client import AttackClient

# from helpers.cifar10 import Cifar10_Helpers
# from helpers.google import Google_Vision_Helpers
# from matplotlib import pyplot as plt

# import json
import argparse

# Example Run Script
# Global Best Server: python3 main.py server global_best -p 3001
# Server Attack: python3 main.py server attack --model_server_url http://172.17.0.3:5000 --global_best_server_url http://localhost:3005 -p 2005 -d mnist -atype untargeted -sb 0.15 -csize 5
# Client Attack: python3 main.py client attack --model_server_url http://172.17.0.3:5000 --global_best_server_url http://localhost:3005 -p 2005 -cid 5_4

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Distributed Black Box Adversarial Attack using MGRR-PSO')

    # - MANDATORY ARGS - #
    # Program Mode
    parser.add_argument('mode', metavar='mode', type=str,
                        help='the program mode [server, client, utils]')
    # Program Type
    parser.add_argument('type', metavar='type', type=str,
                        help='the program type. e.g: server [global_best, model, attack], client [attack], utils []')  # TODO: Add utils

    # - SERVER ARGS - #
    server_arg = parser.add_argument_group('server')
    # Hostname
    server_arg.add_argument('-host', '--hostname',
                            action='store_const', const='0.0.0.0', help='server hostname [optional for server]')
    # Port
    server_arg.add_argument(
        '-p', '--port', help='server port [optional for server and client attack]')

    # - ATTACK ARGS - #
    attack_arg = parser.add_argument_group('attack')

    # Dataset Name
    attack_arg.add_argument(
        '-d', '--dataset', help='dataset that will be attacked')
    # Model Server URL
    attack_arg.add_argument('--model_server_url',
                            default='http://localhost:5000', help='Model Server Base URL')
    # Global Best Server URL
    attack_arg.add_argument('--global_best_server_url',
                            default='http://localhost:6000', help='Global Server URL')
    # Attack Type
    attack_arg.add_argument(
        '-atype', '--attack_type',
        default='untargeted', help='untargeted / targeted attack')
    # Number of Particle
    attack_arg.add_argument(
        '-psize', '--particle_size',
        default=4, help='Number of particle used in MGRR-PSO')
    # Number of Particle
    attack_arg.add_argument(
        '-sb', '--search_boundary',
        default=4, help='Number of particle used in MGRR-PSO')
    # Max Iteration
    attack_arg.add_argument(
        '-iter', '--iteration',
        default=1000, help='Maximum number of iteration')
    # Attack Index
    attack_arg.add_argument(
        '-aidx', '--attack_index',
        default=0, help='Start / Attack Index')
    # Attack Count
    attack_arg.add_argument(
        '-acount', '--attack_count',
        default=1000, help='Batch Attack Count')
    # Number of Client
    attack_arg.add_argument(
        '-csize', '--client_size',
        default=3, help='Number of Attack Clients')
    # Increment Boundary
    attack_arg.add_argument(
        '-inc_sb', type=float, help='Auto increment search boundry when AE searching failed. Should also define max_sb')
    # Max Boundary
    attack_arg.add_argument(
        '-max_sb', default=1.0,  type=float,
        help='Maximum boundary for auto increment sb')

    # - CLIENT ARGS - #
    client_arg = parser.add_argument_group('client')
    # Client Id
    client_arg.add_argument('-cid', '--client_id', help='Client Id')

    args = parser.parse_args()
    print(args)

    # MODE = SERVER
    if args.mode == 'server':

        # Global Best Server
        if args.type == 'global_best':
            from server.global_best_server import app as global_best_server

            # Default param
            host = '0.0.0.0' if args.hostname == None else args.hostname
            port = 6000 if args.port == None else int(args.port)

            global_best_server.run(host=host, port=port, debug=False)

        # Model Server / AI Service Provider
        if args.type == 'model':
            from server.ai_model_server import app as model_server

            # Default param
            host = '0.0.0.0' if args.hostname == None else args.hostname
            port = 5000 if args.port == None else int(args.port)

            model_server.run(host=host, port=port, debug=False)

        if args.type == 'attack':
            from server.attack_server import AttackServer
            from utils.attack_utils import AttackTarget

            # Check Mandatory Param for attack server
            if args.dataset == None or args.attack_type == None or args.search_boundary == None or args.client_size == None:
                print('dataset, attack_type, and search_boundry should be defined')
                exit()

            # Default Param
            port = 2004 if args.port == None else int(args.port)

            attack_server = AttackServer(
                tcp_port=port, model_server_url=args.model_server_url, global_best_server_url=args.global_best_server_url, inc_boundary=args.inc_sb, max_boundary=args.max_sb)
            attack_target = AttackTarget(
                args.dataset, args.attack_type, args.particle_size, args.search_boundary, args.iteration, attack_index=args.attack_index, attack_count=args.attack_count)
            attack_server.run(int(args.client_size), attack_target)

    elif args.mode == 'client':

        # Attack client
        if args.type == 'attack':
            from client.attack_client import AttackClient

            if args.client_id == None:
                print('client_id should be defined')
                exit()
            
            # Default Param
            port = 2004 if args.port == None else int(args.port)
                
            attack_client = AttackClient(
                args.client_id, tcp_port=port, model_server_url=args.model_server_url, global_best_server_url=args.global_best_server_url)
            attack_client.run()


    # mode = sys.argv[1]
    # if mode == 'global_best_server':
    #     global_best_server.run(host='0.0.0.0', port=6000, debug=False)

    # if mode == 'model_server':
    #     model_server.run(host='0.0.0.0', port=5000, debug=False)

    # if mode == 'attack_server':
    #     attack_server = AttackServer(tcp_port=2005, model_server_url=None, global_best_server_url=None)
    #     attack_target = AttackTarget(
    #         'mnist', 'untargeted', 4, 0.25, 1000, attack_index=0, attack_count=1000)
    #     attack_server.run(5, attack_target)

    # if mode == 'attack_client':
    #     client_id = sys.argv[2]
    #     attack_client = AttackClient(
    #         client_id, tcp_port=2005, model_server_url=None, global_best_server_url=None)
    #     attack_client.run()

    # if mode == 'save_global_best':
    #     class_name = sys.argv[2]
    #     if class_name == 'google':
    #         global_best_utils = Global_Best_Utils()
    #         google_helpers = Google_Vision_Helpers()
    #         original_image_path = sys.argv[3]
    #         save_image_path = sys.argv[4]
    #         original_img = google_helpers.load_img(original_image_path)
    #         gbest, gbest_pos = global_best_utils.query_global_best()
    #         img = gbest_pos.reshape(original_img.shape)
    #         plt.imsave(save_image_path, img)
    #     if class_name == 'mnist':
    #         global_best_utils = Global_Best_Utils()
    #         mnist_helpers = Mnist_Helpers()
    #         save_image_path = sys.argv[3]
    #         gbest, gbest_pos = global_best_utils.query_global_best()
    #         img = gbest_pos.reshape((28, 28))
    #         plt.imsave(save_image_path, img, cmap='gist_gray')

    # if mode == 'reset':
    #     global_best_utils = Global_Best_Utils()
    #     global_best_utils.reset_global_best()

    # if mode == 'ui':
    #     interface_name = sys.argv[2]
    #     if interface_name == 'attack':
    #         interface_attack = InterfaceAttack()
    #         interface_attack.open_interface()
    #     elif interface_name == 'server':
    #         inerface_server = InterfaceServer()
    #         inerface_server.open_interface()

    # if mode == 'test':
    #     class_name = sys.argv[2]
    #     if class_name == 'global_best_utils':
    #         global_best_utils = Global_Best_Utils()
    #         print('Reset Global Best', global_best_utils.reset_global_best())
    #         print('Update Global Best', global_best_utils.update_global_best(
    #             50.0, np.arange(10, dtype=np.float32)))
    #         print('Query Global Best', global_best_utils.query_global_best())
    #         print('Update Global Best', global_best_utils.update_global_best(
    #             30.0, np.arange(10, 20, dtype=np.float32)))

    #     if class_name == 'base64':
    #         base64_util = Base64_Utils()
    #         mnist_helpers = Mnist_Helpers()
    #         t = mnist_helpers.mnist_test[1].flatten()
    #         print(t.shape)
    #         np_base64 = base64_util.encode_numpy(t)
    #         print(np_base64)
    #         np_ori = base64_util.decode_numpy(np_base64)
    #         print(np_ori.shape)
    #         print(np.array_equal(t, np_ori))

    #     if class_name == 'query':
    #         print('Query')
    #         dataset = sys.argv[3]
    #         if dataset == 'mnist':
    #             index = int(sys.argv[4])
    #             mnist_helpers = Mnist_Helpers()
    #             mnist_img = mnist_helpers.mnist_test[index]
    #             predictions = mnist_helpers.predict_mnist(mnist_img)
    #             print(predictions)
    #         if dataset == 'load-mnist':
    #             print('Load MNIST')
    #             perturbated_image_path = sys.argv[4]
    #             mnist_helpers = Mnist_Helpers()
    #             perturbated_image = mnist_helpers.load_img(
    #                 perturbated_image_path)
    #             label, pred, score = mnist_helpers.predict_mnist(
    #                 perturbated_image.reshape((28, 28)))
    #             plt.title(label)
    #             plt.imshow(perturbated_image.reshape(
    #                 (28, 28)), cmap='gist_gray')
    #             plt.show()
    #         if dataset == 'cifar10':
    #             index = int(sys.argv[4])
    #             cifar_helpers = Cifar10_Helpers()
    #             cifar_img = cifar_helpers.cifar10_test[index]
    #             predictions = cifar_helpers.predict_cifar10(cifar_img)
    #             print(predictions)

    #     if class_name == 'helpers':
    #         dataset = sys.argv[3]
    #         if dataset == 'google':
    #             google_helpers = Google_Vision_Helpers()
    #             img = google_helpers.load_img(sys.argv[4])
    #             target_label = ' '.join(sys.argv[5:])
    #             print('target_label: ', target_label)
    #             print(img.shape, img.dtype)
    #             results = google_helpers.query_img(img)
    #             print(results)
    #             score, highest = google_helpers.get_score(
    #                 results, target_label)
    #             print('score', score, highest)
            # plt.imshow(img)
            # plt.show()

        # if class_name == 'attack':
        #     test_attack = InterfaceAttack()
        #     test_attack.attack_interface()
        #     dataset = sys.argv[3]
        #     print(class_name, dataset)
        #     if dataset == 'mnist':
        #         attack_type = sys.argv[4]
        #         mnist_attack = MGRR_PSO_ATTACK('mnist', (784,), particle_size=8)
        #         if attack_type == 'untargeted':
        #             test_index = sys.argv[5]
        #             err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred = mnist_attack.untargeted_attack(
        #                 int(test_index), 0.2, max_iteration=1000, auto_stop=True)
        #             label, pred, score = mnist_attack.dataset_helper.predict(
        #                 pos_best.reshape((28, 28)))
        #             plt.title(label)
        #             plt.imshow(pos_best.reshape((28, 28)), cmap='gist_gray')
        #             plt.show()
        #         elif attack_type == 'targeted':
        #             test_index = sys.argv[5]
        #             target_label = sys.argv[6]
        #             print('Generating Adversarial Example of MNIST test index {} into label {}'.format(
        #                 test_index, target_label))
        #             err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred = mnist_attack.targeted_attack(
        #                 int(test_index), int(target_label), 0.2, max_iteration=1000, auto_stop=True)

        #     if dataset == 'cifar10':
        #         attack_type = sys.argv[4]
        #         dataset_shape = (32, 32, 3)
        #         cifar_attack = MGRR_PSO_ATTACK('cifar10', (3072,), particle_size=8)
        #         if attack_type == 'untargeted':
        #             test_index = sys.argv[5]
        #             err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred = cifar_attack.untargeted_attack(
        #                 int(test_index), 0.05, max_iteration=1000, auto_stop=True)
        #             label, pred, score = cifar_attack.dataset_helper.predict(
        #                 pos_best.reshape(dataset_shape), with_softmax=True
        #                 )
        #             plt.title(label)
        #             plt.imshow(pos_best.reshape(dataset_shape))
        #             plt.show()
        #         elif attack_type == 'targeted':
        #             test_index = sys.argv[5]
        #             target_label = sys.argv[6]
        #             print('Generating Adversarial Example of MNIST test index {} into label {}'.format(
        #                 test_index, target_label))
        #             err_best, pos_best, cost_history, iteration, elapsed_time, distances, pred = mnist_attack.targeted_attack(
        #                 int(test_index), int(target_label), 0.2, max_iteration=1000, auto_stop=True)

        #     if dataset == 'google-targeted':
        #         print('Launching Google Targeted Attack')
        #         l_inf = sys.argv[4]
        #         image_path = sys.argv[5]
        #         target_label = ' '.join(sys.argv[6:])
        #         google_attack = Google_Attack(image_path,  particle_size=8)
        #         err_best, pos_best, cost_history, iteration, elapsed_time, distances = google_attack.targeted_attack(
        #             target_label, float(l_inf), max_iteration=50, auto_stop=True)
        #         print(err_best, pos_best, cost_history,
        #               iteration, elapsed_time, distances)
        #         with open('result/' + target_label + '_targeted.json', 'w') as outfile:
        #             json.dump({'err_best': str(err_best), 'pos_best': pos_best.tolist(
        #             ), 'iteration': iteration, 'elapsed_time': elapsed_time, 'distances': str(distances)}, outfile)

        #     if dataset == 'google':
        #         print('Launching Google Attack')
        #         l_inf = sys.argv[4]
        #         image_path = sys.argv[5]
        #         target_label = ' '.join(sys.argv[6:])
        #         google_attack = Google_Attack(image_path, particle_size=8)
        #         err_best, pos_best, cost_history, iteration, elapsed_time, distances = google_attack.untargeted_attack(
        #             target_label, float(l_inf), max_iteration=100, auto_stop=False)
        #         print(err_best, pos_best, cost_history,
        #               iteration, elapsed_time, distances)
        #         with open('result/' + target_label + '.json', 'w') as outfile:
        #             json.dump({'err_best': str(err_best), 'pos_best': pos_best.tolist(
        #             ), 'iteration': iteration, 'elapsed_time': elapsed_time, 'distances': str(distances)}, outfile)

        #     if dataset == 'google-multi':
        #         print('Launching Multi Google Attack')
        #         image_path = 'img/cat.jpg'
        #         target_labels = ['Cat', 'Mammal', 'Felidae']
        #         google_attack = Google_Attack(image_path, particle_size=4)
        #         err_best, pos_best, cost_history, iteration, elapsed_time, distances = google_attack.untargeted_attack_multiple(
        #             target_labels, 0.15, max_iteration=50, auto_stop=False)
        #         print(err_best, pos_best, cost_history,
        #               iteration, elapsed_time, distances)
        #         with open('result/' + '-'.join(target_labels) + '.json', 'w') as outfile:
        #             json.dump({'err_best': str(err_best), 'pos_best': pos_best.tolist(
        #             ), 'iteration': iteration, 'elapsed_time': elapsed_time, 'distances': str(distances)}, outfile)

        # if class_name == 'clean':
        #     dataset = sys.argv[3]
        #     print(class_name, dataset)
            # if dataset == 'mnist':
            #     attack_type = sys.argv[4]
            #     mnist_helpers = Mnist_Helpers()
            #     mnist_attack = MNIST_Attack(particle_size=8)
            #     if attack_type == 'untargeted':
            #         test_index = sys.argv[5]
            #         perturbated_image_path = sys.argv[6]
            #         perturbated_image = mnist_helpers.load_img(
            #             perturbated_image_path)
            #         print(perturbated_image.shape)
            #         results = mnist_attack.untargeted_clean(
            #             int(test_index), perturbated_image,  0.15, max_iteration=100)
            #         print(results)
