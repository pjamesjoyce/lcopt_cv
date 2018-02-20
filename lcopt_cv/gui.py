from tkinter import *
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image
from PIL import ImageTk
import imutils
import numpy as np
#import tkFileDialog
import cv2
import os
from functools import partial
from collections import OrderedDict

from .grab_flow_chart import *
from .heuristics import *

DEFAULT_CONTROLS = [
            {'name': 'threshLevel',
             'type': 'scale',
             'label': 'Threshold level',
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
             'data': {
                'value': 3,
                'min': 0,
                'max': 10,
                'step': 1
             }
            },
            {'name': 'maskThickness',
             'type': 'scale',
             'label': 'line thickness for box mask',
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
             'data': {
                'value': True,
             }
            },
            {'name': 'skipClosing',
             'type': 'checkbox',
             'label': 'Skip closing step (for incomplete boxes)',
             'data': {
                'value': False,
             }
            },
            {'name': 'unstack',
             'type': 'checkbox',
             'label': 'Unstack spurious links',
             'data': {
                'value': False,
             }
            },
            {'name': 'directional',
             'type': 'checkbox',
             'label': 'Use directional heuristic',
             'data': {
                'value': True,
             }
            },
            {'name': 'prefer_linked',
             'type': 'checkbox',
             'label': 'Use prefer linked heuristic',
             'data': {
                'value': False,
             }
            },
            {'name': 'skipDilation',
             'type': 'checkbox',
             'label': 'Skip dilation step for boxes',
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

        self.sm_Image = 245  # 230
        self.lg_Image = 800  # 750

        this_path = os.path.dirname(os.path.realpath(__file__))
        assets = os.path.join(this_path, "assets")
        icon = os.path.join(assets, "icon.ico")

        self.root = Tk()
        self.root.title("Flow chart image processor")
        self.root.iconbitmap(icon)
        
        self.IMAGEPATH = None
        self.IMAGEPROCESSOR_OBJECT = None

        self.controls = controls

        blank = np.full((self.lg_Image, self.lg_Image), 240, dtype=np.uint8)
        blank = Image.fromarray(blank)
        blank_sm = blank.resize((self.sm_Image, self.sm_Image))
        blank_lg = blank.resize((self.lg_Image, self.lg_Image))
        
        blank_sm = ImageTk.PhotoImage(blank_sm)
        blank_lg = ImageTk.PhotoImage(blank_lg)
        
        self.left_frame = Frame(self.root, width=750, height=750)
        self.left_frame.pack(side="left")

        self.middle_frame = Frame(self.root, width=750, height=750)
        self.middle_frame.pack(side="left")

        self.right_frame = Frame(self.root, width=250, height=750)
        self.right_frame.pack(side="left")

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

        btn = Button(self.right_frame, text="Select an image", command=self.select_image)
        btn.pack(side="top", fill="both", expand="yes", padx="10")

        self.reset_btn = Button(self.right_frame, text="Reset defaults", command=self.reset_defaults, state=DISABLED)
        self.reset_btn.pack(side="top", fill="both", expand="yes", padx="10")

        for control in self.controls:
            setattr(self, control['name'], control['data']['value'])
            
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
                setattr(self, controlName, Scale(self.right_frame, from_=from_, to=to, tickinterval=tickinterval, resolution=resolution, orient=HORIZONTAL, state=DISABLED, length=250, command=partial(self.eventHandler, control['name'], controlName)))
                getattr(self, controlName).set(control['data']['value'])
                getattr(self, controlName).pack()

        self.rp_btn = Button(self.right_frame, text="Reprocess image", command=self.reprocess_image, state=DISABLED)
        self.rp_btn.pack(fill="both", expand="yes", padx="10")

        self.generate_btn = Button(self.right_frame, text="Generate LCA model", command= self.generate_model, state=DISABLED)
        self.generate_btn.pack(fill="both", expand="yes", padx=10)                

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
                unstack_pipeline(ip)

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
            if control['type'] == 'scale':
                controlName = '{}Slider'.format(control['name'])
            elif control['type'] == 'checkbox':
                controlName = '{}Check'.format(control['name'])

            getattr(self, controlName).config(state="normal")

            self.rp_btn.config(state="normal")
            self.reset_btn.config(state="normal")

    def reset_defaults(self):

        for control in self.controls:
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
        wizard = Toplevel(height=500, width=800)
        wizard.geometry('800x800')
        wizard.title("Generate LCA model")
        #frame = Frame(wizard, height=500, width=800)
        #button = Button(wizard, text="Dismiss", command=wizard.destroy)
        #button.pack()
        content = LcaWizard(wizard, self.IMAGEPROCESSOR_OBJECT)
        #content.pack()

    def run(self):
        self.root.mainloop()

class LcaWizard(Frame):
    def __init__(self, parent, ip):
        super().__init__(parent)

        self.toplevel = parent

        
       

        self.button_frame = Frame(self, bd=1, relief="raised")
        self.content_frame = Frame(self, bg="blue")

        self.back_button = Button(self.button_frame, text="<< Back", command=self.back)
        self.next_button = Button(self.button_frame, text="Next >>", command=self.next)
        self.finish_button = Button(self.button_frame, text="Finish", command=self.finish)

        self.grid_rowconfigure(0, minsize=750, weight=1)
        self.grid_rowconfigure(1, minsize=50, weight=0)
        self.grid_columnconfigure(0, minsize=800, weight=0)

        self.button_frame.grid(row=1, column=0, sticky=N+S+E+W)  #.pack(side="bottom", fill="x")
        self.content_frame.grid(row=0, column=0, sticky=N+S+E+W)  #pack(side="top", fill="both", expand=True)
        
        self.steps = [NodeStep(self.content_frame, ip), LinkStep(self.content_frame, ip)]
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
        self.show_step(self.current_step + 1)

    def back(self):
        self.show_step(self.current_step - 1)

    def finish(self):
        self.toplevel.destroy()

class NodeStep(Frame):
    def __init__(self, parent, ip):
        super().__init__(parent)

        self.config(bg="white")

        Label(self, text="Hello - from NodeStep").grid(column=0, row=0)

        self.node_frames = []
        self.data = OrderedDict()

        self.grid_rowconfigure(0, minsize=750, weight=1)
        self.grid_columnconfigure(0, minsize=785, weight=1)

        img_size = 150

        scrollsize = img_size * len(ip.nodes)

        yscrollbar = Scrollbar(self)
        yscrollbar.grid(row=0, column=1, sticky=N+S) # pack(side="right", fill="y") 

        canvas = Canvas(self, bd=0, yscrollcommand=yscrollbar.set, scrollregion=(0, 0, 800, scrollsize))

        scroll_frame = Frame(canvas)

        canvas.grid(row=0, column=0, sticky=N+S+E+W)
        #canvas.pack(side="left", fill="both", expand=True)

        yscrollbar.config(command=canvas.yview)

        #img_size = 150
        
        for n, (k, node) in enumerate(ip.nodes.items()):

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
            self.data[n]['nameEntry'] = Entry(control_frame)
            self.data[n]['nameEntry'].grid(column=1, row=0)

            Label(control_frame, text="Type:").grid(column=0, row=1)
            self.data[n]['typeDropDown'] = Entry(control_frame)
            self.data[n]['typeDropDown'].grid(column=1, row=1)
            
            Label(control_frame, text="External link:").grid(column=0, row=2)
            self.data[n]['extLinkEntry'] = Entry(control_frame)
            self.data[n]['extLinkEntry'].grid(column=1, row=2)


            control_frame.grid(column=1, row=0)

            self.node_frames.append(this_frame)

        for n, f in enumerate(self.node_frames):
            f.grid(column=0, row=n)
        
        canvas.create_window((0,0), window=scroll_frame, anchor=N+W)
        #canvas.config(scrollregion=canvas.bbox(ALL))

        
       


class LinkStep(Frame):
    def __init__(self, parent, ip):
        super().__init__(parent)

        self.config(bg="red")

        Label(self, text="Hello - from LinkStep").pack()

if __name__ == "__main__":

    app = ImageGui()

    app.run()