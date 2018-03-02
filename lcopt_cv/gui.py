from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image
from PIL import ImageTk
import imutils
import numpy as np
#import tkFileDialog
import cv2
import os
from functools import partial
from collections import OrderedDict
import json

from .grab_flow_chart import *
from .heuristics import *
from .send_to_lcopt import LcoptWriter

this_path = os.path.dirname(os.path.realpath(__file__))
assets = os.path.join(this_path, "assets")
icon = os.path.join(assets, "icon.ico")

DEFAULT_CONTROLS = [
            {'name': 'threshLevel',
             'type': 'scale',
             'label': 'Threshold level',
             'display': True,
             'data': {
                'value': 115,
                'min': 0,
                'max': 255,
                'step': 1
             }
            },
            {'name': 'boxDilationIterations',
             'type': 'scale',
             'label': 'Number of box dilation iterations',
             'display': True,
             'data': {
                'value': 1,
                'min': 0,
                'max': 10,
                'step': 1
             }
            },
            {'name': 'boxApproxParameter',
             'type': 'scale',
             'label': 'Box approximation parameter',
             'display': False,
             'data': {
                'value': 0.02,
                'min': 0,
                'max': 0.5,
                'step': 0.01
             }
            },
            {'name': 'sizeThreshold',
             'type': 'scale',
             'label': 'Size threshold',
             'display': True,
             'data': {
                'value': 0.2,
                'min': 0,
                'max': 1,
                'step': 0.01
             }
            },
            {'name': 'duplicateThreshold',
             'type': 'scale',
             'label': 'Threshold for duplicate boxes (euclidian distance)',
             'display': True,
             'data': {
                'value': 10,
                'min': 0,
                'max': 200,
                'step': 1
             }
            },
            {'name': 'lineDilateIterations',
             'type': 'scale',
             'label': 'Number of line dilation iterations',
             'display': True,
             'data': {
                'value': 1,
                'min': 0,
                'max': 10,
                'step': 1
             }
            },
            {'name': 'maskThickness',
             'type': 'scale',
             'label': 'line thickness for box mask',
             'display': True,
             'data': {
                'value': 8,
                'min': 0,
                'max': 20,
                'step': 1
             }
            },
            {'name': 'equalizeBackground',
             'type': 'checkbox',
             'label': 'Equalise background',
             'display': True,
             'data': {
                'value': True,
             }
            },
            {'name': 'skipClosing',
             'type': 'checkbox',
             'label': 'Skip closing step (for incomplete boxes)',
             'display': True,
             'data': {
                'value': False,
             }
            },
            {'name': 'unstack',
             'type': 'checkbox',
             'label': 'Unstack spurious links',
             'display': True,
             'data': {
                'value': False,
             }
            },
            {'name': 'directional',
             'type': 'checkbox',
             'label': 'Use directional heuristic',
             'display': True,
             'data': {
                'value': True,
             }
            },
            {'name': 'prefer_linked',
             'type': 'checkbox',
             'label': 'Use prefer linked heuristic',
             'display': True,
             'data': {
                'value': False,
             }
            },
            {'name': 'skipDilation',
             'type': 'checkbox',
             'label': 'Skip dilation step for boxes',
             'display': False,
             'data': {
                'value': False,
             }
            },
            ]


def resize_to_square(image, side_length):
    old_size = image.size
    ratio =  side_length / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    image = image.resize(new_size, Image.ANTIALIAS)
    temp_image = Image.new("RGBA", (side_length, side_length))
    temp_image.paste(image, ((side_length - new_size[0]) // 2, (side_length - new_size[1]) // 2))
    
    return temp_image


def convert_to_tkinter_image(image, size):

    if len(image.shape) > 2:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error:
            pass

    image = Image.fromarray(image)
    image = resize_to_square(image, size)
    image = ImageTk.PhotoImage(image)

    return image


class ImageGui:

    def __init__(self, controls=DEFAULT_CONTROLS):


        #this_path = os.path.dirname(os.path.realpath(__file__))
        #assets = os.path.join(this_path, "assets")
        #icon = os.path.join(assets, "icon.ico")

        self.root = Tk()
        self.root.title("Flow chart image processor")
        self.root.iconbitmap(icon)

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.root.state("zoomed")

        self.root.minsize(width=int(self.screen_width * 0.6), height=int(self.screen_height * 0.6))
        #self.root.maxsize(width=screen_width, height=int(screen_height*0.8))
        #self.root.attributes('-zoomed', True)

        #self.root.geometry = "{}x{}".format(screen_height, screen_width)
        

        self.sm_Image = int(self.screen_height / 4)  #245  # 230
        self.lg_Image = int(self.sm_Image * 2.5)  #800  # 750

        self.IMAGEPATH = None
        self.IMAGEPROCESSOR_OBJECT = None

        self.controls = controls

        blank = np.full((self.lg_Image, self.lg_Image), 240, dtype=np.uint8)
        blank = Image.fromarray(blank)
        blank_sm = blank.resize((self.sm_Image, self.sm_Image))
        blank_lg = blank.resize((self.lg_Image, self.lg_Image))
        
        blank_sm = ImageTk.PhotoImage(blank_sm)
        blank_lg = ImageTk.PhotoImage(blank_lg)
        
        self.left_frame = Frame(self.root)
        self.left_frame.grid(column=0, row=0, rowspan=2)
        #self.left_frame.pack(side="left")

        self.middle_frame = Frame(self.root)
        #self.middle_frame.pack(side="left")
        self.middle_frame.grid(column=1, row=0)

        self.button_frame = Frame(self.root)
        self.button_frame.grid(column=1, row=1)

        self.right_frame = Frame(self.root)
        #self.right_frame.pack(side="left", expand=True)
        self.right_frame.grid(column=2, row=0, rowspan=2)

        # set up image panels

        self.panelNames = [
            "Original image", 
            "Thresholded image", 
            "Dilated/Closed image", 
            "Contours found (boxes in blue)", 
            "Boxes kept",
            "Lines"
            ]

        self.panels = []

        for n, p in enumerate(self.panelNames):
            row = n - (n % 2)
            column = n % 2
            Label(self.left_frame, text=p).grid(column=column, row=row)
            panel = Label(self.left_frame, image=blank_sm, width=self.sm_Image, relief="groove")
            panel.image = blank_sm
            panel.grid(column=column, row=row + 1, padx=10)
            self.panels.append(panel)

        Label(self.middle_frame, text="Result").pack()
        self.panelB = Label(self.middle_frame, image=blank_lg, width=self.lg_Image, relief="groove")
        self.panelB.image = blank_lg
        self.panelB.pack(side="left", padx=10)

        # set up controls

        btn = Button(self.button_frame, text="Select an image", command=self.select_image)
        btn.pack(side="left")

        self.reset_btn = Button(self.button_frame, text="Reset defaults", command=self.reset_defaults, state=DISABLED)
        self.reset_btn.pack(side="left")

        for control in self.controls:
            setattr(self, control['name'], control['data']['value'])

            if control['display']:
            
                if control['type'] == 'checkbox':
                    setattr(self, '{}Var'.format(control['name']), BooleanVar())

                    controlName = '{}Check'.format(control['name'])
                    variableName = '{}Var'.format(control['name'])

                    setattr(self, controlName, Checkbutton(
                        self.right_frame, text=control['label'], variable=getattr(self, variableName),
                        onvalue=True, offvalue=False, command=partial(self.eventHandler, control['name'], variableName), state=DISABLED
                    ))

                    if getattr(self, control['name']):
                        getattr(self, controlName).select()

                    getattr(self, controlName).pack(anchor='w')

                elif control['type'] == 'scale':
                    Label(self.right_frame, text=control['label']).pack()
                    controlName = '{}Slider'.format(control['name'])
                    from_ = control['data']['min']
                    to = control['data']['max']
                    tickinterval = to - from_
                    resolution = control['data']['step']
                    setattr(self, controlName, Scale(
                        self.right_frame, from_=from_, to=to, tickinterval=tickinterval,
                        resolution=resolution, orient=HORIZONTAL, state=DISABLED, length=300, width=10,
                        command=partial(self.eventHandler, control['name'], controlName)
                        ))
                    getattr(self, controlName).set(control['data']['value'])
                    getattr(self, controlName).pack()

        self.rp_btn = Button(self.button_frame, text="Reprocess image", command=self.reprocess_image, state=DISABLED)
        self.rp_btn.pack(side="left")

        self.launch_lcopt_btn = Button(self.button_frame, text="Launch LCA model", command=self.launch_lcopt, state=DISABLED)  
        self.launch_lcopt_btn.pack(side="right")  

        self.generate_btn = Button(self.button_frame, text="Generate LCA model", command= self.generate_model, state=DISABLED)
        self.generate_btn.pack(side="right")

                    

    def select_image(self):

        self.IMAGEPATH = filedialog.askopenfilename()

        self.create_ip_object()

        self.enable_controls()

        self.reset_defaults()

        self.process_image()

    def create_ip_object(self):
        
        self.IMAGEPROCESSOR_OBJECT = ImageProcessor(self.IMAGEPATH)

    def process_image(self):    

        if len(self.IMAGEPATH) > 0:

            image = cv2.imread(self.IMAGEPATH)
            ip = self.IMAGEPROCESSOR_OBJECT
            ip.process(threshLevel=self.threshLevel, 
                       boxApproxParameter=self.boxApproxParameter, 
                       sizeThreshold=self.sizeThreshold, 
                       lineDilateIterations=self.lineDilateIterations, 
                       equalizeBackground=self.equalizeBackground, 
                       skipClosing=self.skipClosing, 
                       maskThickness=self.maskThickness,
                       skipDilation=self.skipDilation,
                       duplicateThreshold=self.duplicateThreshold,
                       boxDilationIterations=self.boxDilationIterations) 
            
            if self.unstack:
                unstack_pipeline(ip, maskThickness = self.maskThickness)

            if self.directional:
                directional_links_pipeline(ip)
            
            if self.prefer_linked:
                prefer_linked_pipeline(ip)

            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #edged = cv2.Canny(gray, 50, 100)
            intermediate_panels = [
                ip.intermediates['original'],
                ip.intermediates['threshold'],
                ip.intermediates['closed'],
                ip.intermediates['contours'],
                ip.intermediates['boxes'],
                ip.intermediates['lines'],
            ]

            for n, panel in enumerate(intermediate_panels):
                image = convert_to_tkinter_image(panel, self.sm_Image)
                self.panels[n].configure(image=image)
                self.panels[n].image = image

            final = convert_to_tkinter_image(ip.intermediates['final'], self.lg_Image)
            self.panelB.configure(image=final)
            self.panelB.image = final

            if len(ip.links) > 0:
                self.generate_btn.config(state="normal")
            else:
                self.generate_btn.config(state="disabled")

    def eventHandler(self, *args):
        #print(args)
        attrName = args[0]
        controlName = args[1]

        setattr(self, attrName, getattr(self, controlName).get())

    def enable_controls(self):

        for control in self.controls:
            if control['display']:
                if control['type'] == 'scale':
                    controlName = '{}Slider'.format(control['name'])
                elif control['type'] == 'checkbox':
                    controlName = '{}Check'.format(control['name'])

                getattr(self, controlName).config(state="normal")

        self.rp_btn.config(state="normal")
        self.reset_btn.config(state="normal")

    def reset_defaults(self):

        for control in self.controls:
            if control['display']:
                setattr(self, control['name'], control['data']['value'])
                if control['type'] == 'scale':
                    getattr(self, '{}Slider'.format(control['name'])).set(control['data']['value'])
                elif control['type'] == 'checkbox':
                    if control['data']['value']:
                        getattr(self, '{}Check'.format(control['name'])).select()
                    else:
                        getattr(self, '{}Check'.format(control['name'])).deselect()

    def reprocess_image(self):
        self.process_image()

    def generate_model(self):
        #messagebox.showinfo("Information", "Informative message")
        w = min(1000, self.screen_width)
        h = min(750, int(self.screen_height*0.8))

        #print(w, h)

        wizard = Toplevel()
        wizard.maxsize(width=w, height=h)

        wizard.geometry("%dx%d%+d%+d" % (w, h, 50, 50))  # int(self.screen_width-w/2), int(self.screen_height-h/2)))
        #wizard.geometry('500x800')
        wizard.title("Generate LCA model")
        wizard.iconbitmap(icon)
        #frame = Frame(wizard, height=500, width=800)
        #button = Button(wizard, text="Dismiss", command=wizard.destroy)
        #button.pack()
        content = LcaWizard(wizard, self.IMAGEPROCESSOR_OBJECT, self, w, h)
        #content.pack()
        #if self.IMAGEPROCESSOR_OBJECT.model is not None:
        #    self.launch_lcopt_btn.config(state="normal")

    def launch_lcopt(self):

        self.root.withdraw()

        model = self.IMAGEPROCESSOR_OBJECT.model

        model.launch_interact()

        self.root.deiconify()

        self.root.state("zoomed")


    def run(self):
        self.root.mainloop()

class LcaWizard(Frame):
    def __init__(self, parent, ip, root, w, h):
        super().__init__(parent)

        self.toplevel = parent

        self.ip = ip

        path, fname = os.path.split(self.ip.imagepath)

        self.lw = LcoptWriter(self.ip, "{}_model".format(fname), False)

        self.ip.model = self.lw.get_model()

        self.root = root

        self.button_frame = Frame(self, bd=1, relief="raised")
        self.content_frame = Frame(self)

        self.back_button = Button(self.button_frame, text="<< Back", command=self.back)
        self.next_button = Button(self.button_frame, text="Next >>", command=self.next)
        self.finish_button = Button(self.button_frame, text="Finish", command=self.finish)

        #self.grid_rowconfigure(0, minsize=450, weight=1)
        #self.grid_rowconfigure(1, minsize=50, weight=0)
        #self.grid_columnconfigure(0, minsize=500, weight=0)

        self.button_frame.grid(row=1, column=0, sticky=N+S+E+W)  #.pack(side="bottom", fill="x")
        self.content_frame.grid(row=0, column=0, sticky=N+S+E+W)  #pack(side="top", fill="both", expand=True)
        
        self.steps = [NodeStep(self.content_frame, ip, w, h), LinkStep(self.content_frame, ip, w, h)]
        self.current_step = 0
        self.show_step(0)

        self.pack(fill="both", expand=True)


    def show_step(self, step):

        if self.current_step is not None:
            #remove currrent step
            current_step = self.steps[self.current_step]
            current_step.grid_forget()

        self.current_step = step

        new_step = self.steps[step]
        new_step.grid(row=0, column=0, sticky=N+S+E+W)  #pack(side="top", fill="both", expand=True)

        if step == 0:
            self.back_button.pack_forget()
            self.next_button.pack(side="right")
            self.finish_button.pack_forget()

        elif step == len(self.steps) - 1:
            self.back_button.pack(side="left")
            self.next_button.pack_forget()
            self.finish_button.pack(side="right")

        else:
            self.back_button.pack(side="left")
            self.next_button.pack(side="right")
            self.finish_button.pack_forget()


    def next(self):

        print(self.ip.model.database['name'])
        print([d['name'] for d in self.ip.model.external_databases])

        if self.current_step == 0:

            senders = [v['link'][0] for k, v in self.ip.links.items()]
            receivers = [v['link'][1] for k, v in self.ip.links.items()]

            for k,v in self.steps[0].data.items():
                self.ip.nodes[k]['name'] = v['nameVar'].get()
                this_type = v['typeVar'].get()
                self.ip.nodes[k]['type'] = this_type

                if this_type in ['input', 'biosphere']:
                    self.ip.nodes[k]['ext_link'] = (v['extLinkDBVar'].get(), v['extLinkCodeVar'].get())

                if self.ip.nodes[k]['type'] == 'input' or self.ip.nodes[k]['type'] == 'biosphere':

                    if k in receivers:
                        print("need to reverse links for item {}, ({})".format(k, self.ip.nodes[k]['name']))
                        links_to_fix = [i for i, l in self.ip.links.items() if l['link'][1] == k]
                        if len(links_to_fix) > 1 :
                            print("too many links - this can't be an input/biosphere exchange")
                        else:
                            old_key = self.ip.links[links_to_fix[0]]['link']
                            new_key = (old_key[1],old_key[0])

                            old_line = self.ip.links[links_to_fix[0]]['centroids']
                            new_line = [old_line[1], old_line[0]]

                            self.ip.links[links_to_fix[0]]['link'] = new_key
                            self.ip.links[links_to_fix[0]]['centroids'] = new_line

        #print(self.ip.links)

        self.steps[1].draw()
        self.show_step(self.current_step + 1)

    def back(self):
        self.show_step(self.current_step - 1)

    def finish(self):

        self.lw.create()

        self.ip.model = self.lw.get_model()

        self.toplevel.destroy()

        self.root.launch_lcopt_btn.config(state="normal")

class NodeStep(Frame):
    def __init__(self, parent, ip, w, h):
        super().__init__(parent)

        self.ip = ip

        #self.config(bg="white")

        # this needs to be transferred to the gui
        senders = [v['link'][0] for k, v in ip.links.items()]
        receivers = [v['link'][1] for k, v in ip.links.items()]

        inputs = [n for n in ip.nodes.keys() if n in senders and n not in receivers]
        intermediates = [n for n in ip.nodes.keys() if n not in inputs]

        types = ['input', 'intermediate', 'biosphere']

        #Label(self, text="Hello - from NodeStep").grid(column=0, row=0)

        self.node_frames = []
        self.data = OrderedDict()

        #self.grid_rowconfigure(0, minsize=450, weight=1)
        #self.grid_columnconfigure(0, minsize=785, weight=1)

        img_size = 150
        scroll_pixels = 20
        panel_pixels = 30

        scrollsize = img_size * len(ip.nodes)

        yscrollbar = Scrollbar(self)
        yscrollbar.grid(row=0, column=1, sticky=N+S) # pack(side="right", fill="y") 

        canvas = Canvas(self, bd=0, yscrollcommand=yscrollbar.set, scrollregion=(0, 0, w, scrollsize))
        canvas.config(width=w-scroll_pixels, height = h-panel_pixels)

        scroll_frame = Frame(canvas)

        canvas.grid(row=0, column=0, sticky=N+S+E+W)
        #canvas.pack(side="left", fill="both", expand=True)

        yscrollbar.config(command=canvas.yview)

        #img_size = 150
        
        for n, (k, node) in enumerate(ip.nodes.items()):


            associated_links = [l for l, v in ip.links.items() if v['link'][0] == k or v['link'][1] == k]

            this_frame = Frame(scroll_frame, width=800, height=img_size)
            image_frame = Frame(this_frame, width=200, height=img_size, relief="groove")
            control_frame = Frame(this_frame, width=600, height=img_size, relief="groove")

            box_image = convert_to_tkinter_image(ip.box_images[n], img_size)

            box_img = Label(image_frame, image=box_image, relief="groove")
            box_img.image = box_image
            box_img.grid(column=0, row=0)

            box_highlight = ip.intermediates['original'].copy()
            (x, y, w, h) = node['coords']
            cv2.rectangle(box_highlight, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=8)
            #cv2.putText(box_highlight, '{}'.format(n), (int(x+5), int(y+h-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

            box_highlight_image = convert_to_tkinter_image(box_highlight, img_size)

            box_highlight_img = Label(image_frame, image=box_highlight_image, relief="groove")
            box_highlight_img.image = box_highlight_image
            box_highlight_img.grid(column=1, row=0)

            image_frame.grid(column=0, row=0)

            self.data[n]={}

            Label(control_frame, text="Name:").grid(column=0, row=0)
            #self.data[n]['name'] = "Box {}".format(n)
            self.data[n]['nameVar'] = StringVar()
            #self.data[n]['nameVar'].trace("w", lambda name, index, mode, sv=self.data[n]['nameVar']: self.updateData(n, sv))
            self.data[n]['nameVar'].set("Box {}".format(n+1))
            self.data[n]['nameEntry'] = Entry(control_frame, textvariable=self.data[n]['nameVar'])
            self.data[n]['nameEntry'].grid(column=1, row=0)

            Label(control_frame, text="Type:").grid(column=0, row=1)
            self.data[n]['typeVar'] = StringVar()
            
            if n in inputs:
                this_type = 'input'
            else:
                this_type = 'intermediate'

            self.data[n]['typeVar'].set(this_type)
            if len(associated_links) > 1:
                self.data[n]['typeDropDown'] = ttk.Combobox(control_frame, textvariable=self.data[n]['typeVar'], values=['intermediate'])  # OptionMenu(control_frame, self.data[n]['typeVar'], 'intermediate')#, state=DISABLED) 
            else:
                self.data[n]['typeDropDown'] = ttk.Combobox(control_frame, textvariable=self.data[n]['typeVar'], values=types)  #OptionMenu(control_frame, self.data[n]['typeVar'], *(types), command=partial(self.changeType, n)) 
                self.data[n]['typeDropDown'].bind("<<ComboboxSelected>>", partial(self.changeType, n))
                #self.data[n]['typeDropDown'].config(command=partial(self.changeType, n))
            self.data[n]['typeDropDown'].grid(column=1, row=1)
            
            search_frame = Frame(control_frame)
            Label(search_frame, text="External link:").grid(column=0, row=0)
            self.data[n]['extLinkVar'] = StringVar()
            self.data[n]['extLinkCodeVar'] = StringVar()
            self.data[n]['extLinkDBVar'] = StringVar()
            self.data[n]['extLinkEntry'] = Entry(search_frame, state=DISABLED, textvariable=self.data[n]['extLinkVar'])
            self.data[n]['extLinkEntry'].grid(column=1, row=0)
            self.data[n]['extLinkbtn'] = Button(search_frame, text='Select', command= partial(self.searchExternal, n))
            
            if this_type == "intermediate":
                self.data[n]['extLinkbtn'].config(state=DISABLED)

            self.data[n]['extLinkbtn'].grid(column=2, row=0)
            search_frame.grid(column=0, row=2, columnspan=2)


            control_frame.grid(column=1, row=0)

            self.node_frames.append(this_frame)

        for n, f in enumerate(self.node_frames):
            f.grid(column=0, row=n)
        
        canvas.create_window((0,0), window=scroll_frame, anchor=N+W)
        #canvas.config(scrollregion=canvas.bbox(ALL))

    def changeType(self, *args):
        #print(args)
        this_id = args[0]
        #this_type = args[1]
        #print(this_id, this_type)
        this_type = self.data[this_id]['typeVar'].get()
        print(this_id, this_type)
        if this_type in ['input', 'biosphere']:
            self.data[this_id]['extLinkbtn'].config(state=NORMAL)
        else:
            self.data[this_id]['extLinkbtn'].config(state=DISABLED)
            self.data[this_id]['extLinkVar'].set('')


    def searchExternal(self, this_id):
        this_type = self.data[this_id]['typeVar'].get()
        
        ext_databases = [x['name'] for x in self.ip.model.external_databases]

        if this_type == 'input':
            to_search = [x for x in ext_databases if x in self.ip.model.technosphere_databases]
        else:
            to_search = [x for x in ext_databases if x in self.ip.model.biosphere_databases]

        print(this_id, this_type)
        print(to_search)

        searcher = DataSearcher(self, self.ip.model, to_search, this_type, self.data[this_id]['extLinkVar'], self.data[this_id]['extLinkCodeVar'], self.data[this_id]['extLinkDBVar'])

        searcher.show()


class LinkStep(Frame):
    def __init__(self, parent, ip, w, h):
        super().__init__(parent)

        self.parent = parent
        self.ip = ip

        self.data = OrderedDict()

        self.image_frame = Frame(self)
        self.image_frame.grid(column=0,row=0, sticky=N+S)

        self.w = w
        self.h = h

        scroll_pixels = 20
        panel_pixels = 30

        scrollsize = 800

        self.img_size = int(min((w - scroll_pixels) * 0.6, h-panel_pixels))

        yscrollbar = Scrollbar(self)
        yscrollbar.grid(row=0, column=2, sticky=N+S) # pack(side="right", fill="y") 

        control_width = w - self.img_size - scroll_pixels - 5

        self.canvas = Canvas(self, bd=0, yscrollcommand=yscrollbar.set, scrollregion=(0, 0, w - scroll_pixels, scrollsize))
        self.canvas.config(width=control_width, height = h - panel_pixels)

        self.canvas.grid(row=0, column=1, sticky=N+S+E+W)

        yscrollbar.config(command=self.canvas.yview)

        self.scroll_frame = Frame(self.canvas)

        self.control_frame = Frame(self.scroll_frame)
        self.control_frame.grid(row=0, column=0, sticky=N+S+E+W)
       

        self.canvas.create_window((0,0), window=self.scroll_frame, anchor=N+W)
        

    def draw(self):

        intermediates = [k for k, v in self.ip.nodes.items() if v['type'] =='intermediate']

        int_to_int_links = [l for l, v in self.ip.links.items() if v['link'][0] in intermediates and v['link'][1] in intermediates]

        #print(int_to_int_links)

        control_width = self.w - self.img_size - 25



        #clear control_frame

        for i in self.control_frame.grid_slaves():
            i.destroy()

        link_frames = []

        Label(self.control_frame, text="Click on the arrow buttons to reverse the link").grid(column=0, row=0)

        for link in int_to_int_links:

            link_tuple = self.ip.links[link]['link']

            self.data[link] = {}
            self.data[link]['linkVar'] = BooleanVar()
            self.data[link]['linkVar'].set(False)

            this_frame = Frame(self.control_frame)

            this_frame.grid_columnconfigure(0, minsize=int(control_width * 0.4))
            this_frame.grid_columnconfigure(1, minsize=int(control_width * 0.2))
            this_frame.grid_columnconfigure(2, minsize=int(control_width * 0.4))

            fromLabel = Label(this_frame, text=self.ip.nodes[link_tuple[0]]['name'])
            self.data[link]['linkBtn'] = Button(this_frame, text="-->", command=partial(self.flip_link, link, self.ip) )
            toLabel = Label(this_frame, text=self.ip.nodes[link_tuple[1]]['name'])

            fromLabel.grid(column=0, row=0)
            self.data[link]['linkBtn'].grid(column=1, row=0)
            toLabel.grid(column=2, row=0)

            link_frames.append(this_frame)

        for n, frame in enumerate(link_frames):
            frame.grid(column=0, row=n+1, pady=10)



        types = ['input', 'intermediate', 'biosphere']
        type_colors = {
            'input': (191, 144, 99),
            'intermediate': (228, 227, 225),
            'biosphere': (102, 96, 91)
        }
        type_line_colors = {
            'input': (154, 111, 46),
            'intermediate': (170, 170, 170),
            'biosphere': (81, 76, 72)
        }
        type_text_colors = {
            'input': (255, 255, 255),
            'intermediate': (0, 0, 0),
            'biosphere': (255, 255, 255)
        }

        #scroll_frame = Frame(self.canvas)

        #img_size = 150

        img_w, img_h = self.ip.image.shape[:2]

        link_image = np.full((img_w, img_h, 3), 255, dtype=np.uint8)
        
        for n, (k, node) in enumerate(self.ip.nodes.items()):
            (x, y, w, h) = node['coords']
            cv2.rectangle(link_image, (x, y), (x + w, y + h), color=type_colors[node['type']], thickness=-1)
            cv2.rectangle(link_image, (x, y), (x + w, y + h), color=type_line_colors[node['type']], thickness=2)
            cv2.putText(link_image, '{}'.format(node['name']), (int(x+15), int(y+h-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, type_text_colors[node['type']], thickness=1)

        for n, (k, link) in enumerate(self.ip.links.items()):

            link_type = self.ip.nodes[link['link'][0]]['type']
            c = link['centroids']

            if link_type != 'biosphere':
                
                x1 = c[0][0]  
                y1 = c[0][1]  
                x2 = c[1][0]  
                y2 = c[1][1]  

            else:

                x2 = c[0][0]  
                y2 = c[0][1]  
                x1 = c[1][0]  
                y1 = c[1][1]  

            
            cv2.arrowedLine(link_image, (x1, y1), (x2, y2), type_line_colors[link_type], thickness=2)


        link_image_tk = convert_to_tkinter_image(link_image, self.img_size)

        show_img = Label(self.image_frame, image=link_image_tk)
        show_img.image = link_image_tk
        show_img.grid(column=0, row=0)


        #for n, f in enumerate(self.link_frames):
        #    f.grid(column=0, row=n)
        
        
        #canvas.config(scrollregion=canvas.bbox(ALL))

    def flip_link(self, *args):
        
        link = args[0]
        ip = args[1]
        
        old_key = self.ip.links[link]['link']
        new_key = (old_key[1],old_key[0])

        old_line = self.ip.links[link]['centroids']
        new_line = [old_line[1], old_line[0]]

        self.ip.links[link]['link'] = new_key
        self.ip.links[link]['centroids'] = new_line

        self.draw()


class DataSearcher(Toplevel):

    def __init__(self, parent, model, to_search, this_type, nameVar, codeVar, dbVar):
        super().__init__(parent)
        self.model = model
        self.parent = parent
        self.to_search = to_search
        self.this_type = this_type
        self.w = 400
        self.h = 400
        self.title('Search for external data in {} database'.format(to_search[0]))
        self.geometry("%dx%d%+d%+d" % (self.w, self.h, 50, 50))

        self.nameVar = nameVar
        self.codeVar = codeVar
        self.dbVar = dbVar


    def show(self):
        Label(self, text="Look for something").pack()
        
        location_path = os.path.join(assets, 'locations.json')

        with open(location_path, 'r', encoding='utf-8') as f:
            locations = json.load(f)

        all_items = [x['items'] for x in self.model.external_databases if x['name'] in self.model.technosphere_databases]

        used_locations = set([x['location'] for item in all_items for _, x in item.items()])

        filtered_locations = [x for x in locations if x['code'] in used_locations]

        self.location_list = {"[{}] {}".format(x['code'], x['name']): x['code'] for x in filtered_locations}

        searchTerm = StringVar()
        enter = Entry(self, textvariable=searchTerm)
        enter.pack()

        marketsOnly = BooleanVar()
        locationVar = StringVar()
        if self.this_type == 'input':
            
            marketsCheck = Checkbutton(self, text="Markets only?", variable=marketsOnly)
            marketsCheck.pack()
            
            locationList = ttk.Combobox(self, textvariable=locationVar, values=list(self.location_list.keys()))
            locationList.pack()

        btn = Button(self, text='Search', command=partial(self.search, searchTerm, marketsOnly, locationVar))
        btn.pack()
        self.result_box = Listbox(self, width=self.w-20)
        self.result_box.pack()

        OKbtn = Button(self, text='OK', command=partial(self.choose))
        OKbtn.pack()

        self.focus_set()
        self.grab_set()
        self.transient(self.parent)
        self.wait_window(self)

    def search(self, *args):
        
        searchTerm = args[0].get()
        marketsOnly = args[1].get()
        location = args[2].get()
        print(searchTerm, marketsOnly, location)
        print (self.to_search)
        
        if location == "":
            location = None
        else:
            location = self.location_list[location]

        if self.this_type == 'input':

            result = self.model.search_databases(searchTerm, databases_to_search=self.to_search, markets_only=marketsOnly, location=location,)

        else:

            result = self.model.search_databases(searchTerm, databases_to_search=self.to_search)  #, markets_only=marketsOnly, location=location,)

        if self.this_type == 'input':
            self.result_as_dict = {"{} [{} {{{}}}] ({})".format(v['reference product'], v['name'], v['location'], v['unit']) : (v['database'], v['code']) for k, v in result.items()}
        else:
            self.result_as_dict = {}

            for k, v in result.items():

                print(k, v.keys())

                if v['type'] == 'emission':
                    full_link_string = '{} (emission to {}) [{}]'.format(v['name'], ", ".join(v['categories']), v['unit'])
                    print(full_link_string)  
                else:
                    full_link_string = '{} ({}) [{}]'.format(v['name'], ", ".join(v['categories']), v['unit'])
                    print(full_link_string) 

                self.result_as_dict[full_link_string] = (v['database'], v['code'])


        self.result_box.delete(0, END)

        for r in self.result_as_dict.keys():
            print(r)
            self.result_box.insert(END, r)

    def choose(self, *args):

        chosen = self.result_box.get(ACTIVE)

        self.nameVar.set(chosen)
        self.dbVar.set(self.result_as_dict[chosen][0])
        self.codeVar.set(self.result_as_dict[chosen][1])

        self.destroy()




if __name__ == "__main__":

    app = ImageGui()

    app.run()