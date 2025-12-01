from typing import List, Callable, Dict, Tuple, Optional
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import print_build

BuildDict = Dict[str, Tuple[str,float]]
json_path = 'data/part1_JSON/parts.json'
#TODO: Implement Motherboard build on main()

def load_parts(path: str)->Dict[str,float]:
    """Read the JSON file in 'path' and return its Dict."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def pick_part(category: str, item: str, parts: Dict[str, float], build: BuildDict)->None:
    """Search in Dict of parts for a item in a category and returns the updated build."""
    if category not in parts:
        raise ValueError(f'Invalid category: {category}')
    if item not in parts[category]:
        raise ValueError(f'Item {item} don\'t exist in category {category}')
    
    build[category] = (item, parts[category][item])

def make_branch(question: str, choices: Optional[List[str]] = None)-> Callable[[], str]:
    """
    Get the question and the list of choices.
    Returns a function that, when it's called, shows the questions and return the selected option.
    """
    if choices is None:
        choices = ['yes', 'y', 'no', 'n']
        show_combined = True
    else:
        show_combined = False
    normalized_choices = [c.lower().strip() for c in choices]
    def step()->str:
        while True:
            print(question)
            if show_combined:
                print(' - yes (y)')
                print(' - no (n)')
            else:
                for c in choices:
                    print(' -', c)
            ans: str = input('> '.lower().strip())
            if ans in normalized_choices:
                return ans
            print('Invalid answer. Pick a valid option!\n')
    return step

# def print_build(build: BuildDict)->None:
#     print('\n'+('='*29)+' FINAL BUILD '+('='*29))
#     print(f"{'Component':20} | {'Choice':35} | Price (R$)")
#     print('-' * 71)

#     total = 0
#     for comp, (name, price) in build.items():
#         total += price
#         print(f'{comp:20} | {name:35} | {price:>8}')

#     print('-' * 71)
#     print(f"{'TOTAL':20} | {'':35} | {total:>8}")
#     print(('='*71)+'\n')

def main()->None:
    build: BuildDict = {}
    afirmative = ['yes', 'y']

    parts = load_parts(json_path)
    #================ PERIPHERALS ================
    ask_controller = make_branch('Do you play with a controller?')
    if ask_controller() in afirmative:
        pick_part('Controller','8Bitdo Ultimate 2',parts,build)
    else:
        pick_part('Controller','No controller',parts,build)

    ask_keyboard = make_branch('Do you do a lot of typing?')
    if ask_keyboard() in afirmative:
        pick_part('Keyboard','Kaihl Brown Keyboard',parts,build)
    else:
        pick_part('Keyboard','Basic Keyboard',parts,build)
    
    ask_camera = make_branch('Do you stream or record yourself?')
    if ask_camera() in afirmative:
        pick_part('Camera','USB 4K Camera',parts,build)
    else:
        pick_part('Camera','No camera',parts,build)

    #================ USED PARTS ================
    ask_used = make_branch('Willing to buy used parts?')
    used = ask_used() =='yes'

    #================ BRANCH: USED ================
    gpu = True
    if used:
        ask_tier = make_branch(
            'Which is the budget tier?',
            ['low','mid','high']
        )
        tier = ask_tier()
        match tier:
            case 'low':
                ask_gpu = make_branch('Do you need GPU?')
                pick_part('CPU','Xeon 2680v4',parts,build)
                pick_part('RAM','16GB DDR4',parts,build)
                if ask_gpu() in afirmative:
                    pick_part('GPU','RX 580',parts,build)
                else:
                    pick_part('GPU','GT 750 Ti',parts,build)
            case 'mid':
                ask_gpu = make_branch('Do you need GPU?')
                pick_part('CPU','Ryzen 5700X3D',parts,build)
                pick_part('RAM','32GB DDR4',parts,build)
                if ask_gpu() in afirmative:
                    pick_part('GPU','RX 6700',parts,build)
                else:
                    pick_part('GPU','RX 580',parts,build)
            case 'high':
                ask_gpu = make_branch('Do you need GPU?')
                pick_part('CPU','Ryzen 7800X3D',parts,build)
                pick_part('RAM','64GB DDR4',parts,build)
                if ask_gpu() in afirmative:
                    pick_part('GPU','RTX 3090',parts,build)
                else:
                    pick_part('GPU','RX 6700',parts,build)
        
    #================ BRANCH: NEW ================
    else:
        ask_tier = make_branch(
            'Which is the budget tier?',
            ['low','mid','high']
        )
        tier = ask_tier()
        match tier:
            case 'low':
                ask_gpu = make_branch('Do you need GPU?')
                gpu = ask_gpu()
                pick_part('RAM','16GB DDR4',parts,build)
                if gpu in afirmative:
                    pick_part('CPU','I5 12600F',parts,build)
                    pick_part('GPU','RX 6600',parts,build)
                else:
                    pick_part('CPU','Ryzen 5600G (APU)',parts,build)
                    pick_part('GPU','Vega 8',parts,build)
            case 'mid':
                ask_gpu = make_branch('Do you need GPU?')
                pick_part('CPU','Ryzen 7600',parts,build)
                pick_part('RAM','32GB DDR5',parts,build)
                if ask_gpu() in afirmative:
                    pick_part('GPU','RTX 4070',parts,build)
                else:
                    pick_part('GPU','RX 6600',parts,build)
            case 'high':
                ask_gpu = make_branch('Do you need GPU?')
                pick_part('CPU','Ryzen 9800X3D',parts,build)
                pick_part('RAM','64GB DDR5',parts,build)
                if ask_gpu() in afirmative:
                    pick_part('GPU','RTX 4090',parts,build)
                else:
                    pick_part('GPU','RTX 4070',parts,build)

    #================ Power Supply Unit (Font) ================
    ask_upgrade = make_branch('Do you intend on upgrading this PC in the future?')
    if ask_upgrade() in afirmative:
        match tier:
            case 'low':
                if not used and not gpu:
                    pick_part('PSU','Corsair CX650W',parts,build)
                else:
                    pick_part('PSU','G700W',parts,build)
            case 'mid':
                pick_part('PSU','G800W',parts,build)
            case 'high':
                pick_part('PSU','GX1000W',parts,build)
    else:
        match tier:
            case 'low':
                if not used and not gpu:
                    pick_part('PSU','Mancer 400W',parts,build)
                else:
                    pick_part('PSU','Corsair CV550W',parts,build)
            case 'mid':
                pick_part('PSU','Corsair CX650W',parts,build)
            case 'high':
                pick_part('PSU','SL850W',parts,build)

    #================ MONITOR ================
    ask_entertainment = make_branch('Do you intend on watching entertainment in this PC?')
    entertainment = ask_entertainment()

    ask_competitive = make_branch('Do you play competitive games?')
    if ask_competitive() in afirmative:
        if entertainment:
            pick_part('Audio','Soundbar',parts,build)
            if tier == 'high':
                pick_part('Monitor','29\" 4K HRR',parts,build)
                pick_part('Mouse','Deathadder V2',parts,build)
            elif tier == 'low':
                pick_part('Mouse','Deathadder Essential',parts,build)
                if not used and not gpu:
                    pick_part('Monitor','27\" 1080p 144Hz',parts,build)
                else:
                    pick_part('Monitor','27\" 1440p 144Hz',parts,build)
            else:
                pick_part('Mouse','Deathadder V2 Mini',parts,build)
                pick_part('27\" 1440p 144Hz',parts,build)
        else:
            pick_part('Audio','None',parts,build)
            if tier == 'high':
                pick_part('Monitor','25\" 4K HRR',parts,build)
                pick_part('Mouse','Deathadder V2',parts,build)
            elif tier == 'low':
                pick_part('Mouse','Deathadder Essential',parts,build)
                if not used and not gpu:
                    pick_part('Monitor','23\" 1080p 144Hz',parts,build)
                else:
                    pick_part('Monitor','24\" 1440p 144Hz',parts,build)
            else:
                pick_part('Mouse','Deathadder V2 Mini',parts,build)
                pick_part('24\" 1440p 144Hz',parts,build)
    else:
        pick_part('Mouse','Basic Mouse',parts,build)
        if entertainment:
            pick_part('Audio','Soundbar',parts,build)
            if tier == 'high':
                pick_part('Monitor','29\" 4K 60Hz',parts,build)
            elif tier == 'low':
                if not used and not gpu:
                    pick_part('Monitor','27\" 1080p 60Hz',parts,build)
                else:
                    pick_part('Monitor','27\" 1440p 60Hz',parts,build)
            else:
                pick_part('27\" 1440p 60Hz',parts,build)
        else:
            pick_part('Audio','None',parts,build)
            if tier == 'high':
                pick_part('Monitor','25\" 4K 60Hz',parts,build)
            elif tier == 'low':
                if not used and not gpu:
                    pick_part('Monitor','23\" 1080p 60Hz',parts,build)
                else:
                    pick_part('Monitor','24\" 1440p 60Hz',parts,build)
            else:
                pick_part('24\" 1440p 144Hz',parts,build)
    
    #================ WIFI ================
    ask_wifi = make_branch('Do you have access to an ethernet cable for the PC?')
    if ask_wifi() in afirmative:
        pick_part('Wifi','Bluetooth Dongle',parts,build)
    else:
        pick_part('Wifi','PCIe Wi-Fi AX200',parts,build)

    #================ FINAL PRINT ================
    print_build(build)

#================ EXECUTION ================
if __name__ == "__main__":
    main()