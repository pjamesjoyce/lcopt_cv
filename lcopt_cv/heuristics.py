import cv2
import itertools
from operator import itemgetter
import numpy as np
from itertools import combinations


def round_down(num, divisor):
    return num - (num % divisor)


def nodes_as_dict(ip):
    return [{'index': n, 'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h} for n, (x, y, w, h) in enumerate(ip.box_coords)]


def get_stacks(node_dict, links):

    sorted_node_dict = sorted(node_dict, key=itemgetter('x1'))

    stacks = []

    for key, group in itertools.groupby(sorted_node_dict, key=lambda x: round_down(x['x1'], 5)):
        l = int(key)
        g = list(group)
        
        right_sorted_node_dict = sorted(g, key=itemgetter('x2'), reverse=True)
        r = right_sorted_node_dict[0]['x2']
        
        top_sorted_node_dict = sorted(g, key=itemgetter('y1'))
        t = top_sorted_node_dict[0]['y1']
        
        bottom_sorted_node_dict = sorted(g, key=itemgetter('y2'), reverse=True)
        b = bottom_sorted_node_dict[0]['y2']
        
        nodes_in_stack = [x['index'] for x in g]
        links_in_stack = [l for l in links.keys() if l[0] in nodes_in_stack and l[1] in nodes_in_stack]
        
        if len(list(g)) > 1:
            stacks.append({'l': l, 'r': r, 't': t, 'b': b, 'data': g, 'nodes': nodes_in_stack, 'links': links_in_stack})

    return stacks


def unstack(ip, stacks, maskThickness = 8):
    
    links_to_drop = []

    for stack in stacks:
        img = ip.intermediates['lines'].copy()
        stack_mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)
        stack_mask[stack['t']:stack['b'], stack['l']:stack['r']] = 255
        temp_mask = cv2.bitwise_and(img, img, mask = stack_mask)

        nodes = ip.box_coords

        linked_processes = {}

        for i in combinations ([x for x in range(len(nodes))], 2):

            centroids = []

            for j in range(2):
                (x1, y1, w, h) = nodes[i[j]]
                x2 = x1 + w
                y2 = y1 + h
                pt1 = (x1, y1)
                pt2 =  (x2, y2)
                ct = (int(x1 + w / 2), int(y1 + h / 2))

                centroids.append(ct)

                color = (255, 255, 255)
                cv2.rectangle(temp_mask, pt1, pt2, color, thickness=-1, lineType=8, shift=0) 
                cv2.rectangle(temp_mask, pt1, pt2, color, thickness=maskThickness, lineType=8, shift=0)

            flood_colour = (127, 127, 127)
            cv2.floodFill(temp_mask, None, centroids[0], flood_colour)

            if temp_mask[centroids[1][1], centroids[1][0]] == 127:

                linked_processes[i] = centroids

        links_to_drop.extend([l for l in stack['links'] if l not in linked_processes.keys()])

    links = {k: v for k, v in ip.links.items() if k not in links_to_drop} 

    return links


def unstack_pipeline(ip, maskThickness=8):

    node_dict = nodes_as_dict(ip)

    stacks = get_stacks(node_dict, ip.links)

    links = unstack(ip, stacks, maskThickness=maskThickness)

    ip.links = links

    ip.redraw_links()

    return None


def directional_links(links, roundTolerance=20):

    directed_links = {}
    
    for l, c in links.items():
        x1 = round_down(c[0][0], roundTolerance)
        x2 = round_down(c[1][0], roundTolerance)
        y1 = round_down(c[0][1], roundTolerance)
        y2 = round_down(c[1][1], roundTolerance)
        
        #print(x1, x2, y1, y2)
        
        if x1 > x2:
            directed_links[(l[1], l[0])] = [c[1], c[0]]
        elif x1 == x2 and y1 > y2:
                directed_links[(l[1], l[0])] = [c[1], c[0]]
        else:
            directed_links[l] = c

    return directed_links


def directional_links_pipeline(ip, roundTolerance=20):
    links = directional_links(ip.links, roundTolerance=20)
    ip.links = links
    ip.redraw_links()


def prefer_linked(links):
    node_link_list = []
    for l in links.keys():
        node_link_list.extend(list(l))

    node_instances = {}
    for key, group in itertools.groupby(sorted(node_link_list)):
        node_instances[key] = len(list(group))

    pl_links = {}

    for l, c in links.items():
        
        if node_instances[l[0]] > 1 and node_instances[l[1]] == 1:
            pl_links[(l[1], l[0])] = [c[1], c[0]]
        else:
            pl_links[l] = c

    return pl_links


def prefer_linked_pipeline(ip):
    links = prefer_linked(ip.links)
    ip.links = links
    ip.redraw_links()
