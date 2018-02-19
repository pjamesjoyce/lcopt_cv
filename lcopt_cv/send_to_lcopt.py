from lcopt import *


def scale_x(ip, val):
    w = ip.image.shape[1]
    return val / w * 800


def scale_y(ip, val):
    w = ip.image.shape[0]
    return val / w * 500


class LcoptWriter:

    def __init__(self, ip, modelname='model', launch=False):

        senders = [k[0] for k in ip.links.keys()]
        receivers = [k[1] for k in ip.links.keys()]

        inputs = [n for n in ip.nodes.keys() if n in senders and n not in receivers]
        intermediates = [n for n in ip.nodes.keys() if n not in inputs]

        model = LcoptModel(modelname)

        for i in intermediates:
            process_name = ip.nodes[i]['name']
            output_name = "Output of {}".format(process_name)
            my_exchanges = [{'name': output_name, 'type': 'production', 'unit': 'kg', 'lcopt_type': 'intermediate'}]
            things_to_link = [x[0] for x in ip.links.keys() if x[1] == i]
            
            for l in things_to_link:
                if l in inputs:
                    this_exchange = {'name': ip.nodes[l]['name'], 'type': 'technosphere', 'unit': 'kg', 'lcopt_type': 'input'}
                    my_exchanges.append(this_exchange)
                else:
                    this_exchange = {'name': "Output of {}".format(ip.nodes[l]['name']), 'type': 'technosphere', 'unit': 'kg', 'lcopt_type': 'intermediate'}
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

        if launch:
            model.launch_interact()
