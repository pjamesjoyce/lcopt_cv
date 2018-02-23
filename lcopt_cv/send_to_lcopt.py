from lcopt import *


def scale_x(ip, val):
    w = ip.image.shape[1]
    return val / w * 800


def scale_y(ip, val):
    w = ip.image.shape[0]
    return val / w * 500


class LcoptWriter:

    def __init__(self, ip, modelname='model', launch=False):

        self.ip = ip
        self.modelname = modelname
        self.launch = launch

        self.model = LcoptModel(modelname)

    def create(self, ip=None, model=None):

        if model is None:
            model = self.model

        if ip is None:
            ip = self.ip

        inputs = [k for k, v in ip.nodes.items() if v['type'] == 'input']
        intermediates = [k for k, v in ip.nodes.items() if v['type'] == 'intermediate']
        biosphere = [k for k, v in ip.nodes.items() if v['type'] == 'biosphere']
        

        for i in intermediates:
            process_name = ip.nodes[i]['name']
            output_name = "Output of {}".format(process_name)
            my_exchanges = [{'name': output_name, 'type': 'production', 'unit': 'kg', 'lcopt_type': 'intermediate'}]
            things_to_link = [x['link'][0] for k, x in ip.links.items() if x['link'][1] == i]
            
            for l in things_to_link:
                if l in inputs or l in biosphere:
                    print(ip.nodes[l]['ext_link'])
                    this_exchange = {'name': ip.nodes[l]['name'], 'type': 'technosphere', 'unit': 'kg', 'lcopt_type': ip.nodes[l]['type'], 'ext_link': ip.nodes[l]['ext_link']}
                    my_exchanges.append(this_exchange)
                else:
                    this_exchange = {'name': "Output of {}".format(ip.nodes[l]['name']), 'type': 'technosphere', 'unit': 'kg', 'lcopt_type': ip.nodes[l]['type']}
                    my_exchanges.append(this_exchange)
                    
            model.create_process(process_name, my_exchanges)

        for k, v in ip.nodes.items():
            (x, y, _w, _h) = v['coords']
            if k in intermediates:
                uuid = model.get_exchange(v['name'])[1]
                model.sandbox_positions[uuid] = {'x': scale_x(ip, x), 'y': scale_y(ip, y)}
            else:
                uuid = '{}__0'.format(model.get_exchange(v['name'])[1])
                model.sandbox_positions[uuid] = {'x': scale_x(ip, x), 'y': scale_y(ip, y)}

        model.save()

        if self.launch:
            model.launch_interact()

        self.model = model

    def get_model(self):
        return self.model
