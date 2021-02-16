from server.global_best_server import app as global_best_server
from server.ai_model_server import app as model_server

from PyInquirer import prompt


class InterfaceServer:

    def __init__(self):
        pass

    def open_interface(self):
        questions = [
            {
                'type': 'list',
                'name': 'server',
                'message': 'Which server ?',
                'choices': [
                    'model_server',
                    'global_best_server',
                ]
            },
        ]

        answers = prompt(questions)
        if answers['server'] == 'model_server':
            model_server.run(host='0.0.0.0', port=5000, debug=False)
        elif answers['server'] == 'global_best_server':
            global_best_server.run(host='0.0.0.0', port=6000, debug=False)
