from typing import List, Callable, Dict, Tuple

BuildDict = Dict[str, Tuple[str,float]]

def ask(prompt):
    """Asks user a yes or no question and returns the answer."""
    while True:
        a = input(f"{prompt} (sim/não): ").strip().lower()
        if a in {"sim","s"}: return "sim"
        if a in {"nao","não","n"}: return "nao"
        print("Use 'sim' ou 'não'.")

def make_branch(question: str, choices: List[str])-> Callable[[], str]:
    """
    Get the question and the list of choices.
    Returns a function that, when it's called, shows the questions and return the selected option.
    """
    normalized_choices = [c.lower().strip() for c in choices]
    def step()->str:
        while True:
            print(question)
            for c in choices:
                print(' -', c)
            ans: str = input('> '.lower().strip())
            if ans in normalized_choices:
                return ans
            print('Invalid answer. Pick a valid option!\n')
    return step

def print_build(build: BuildDict)->None:
    print('\n==================== FINAL BUILD ====================')
    print(f"{'Component':20} | {'Choice':35} | Price (R$)")
    print('-' * 70)

    total = 0
    for comp, (name, price) in build.items():
        total += price
        print(f'{comp:20} | {name:35} | {price:>8}')

    print('-' * 70)
    print(f"{'TOTAL':20} | {'':35} | {total:>8}")
    print('============================================================\n')

def main()->None:
    build: BuildDict = {}
    standard_choices = ['yes', 'no']

    #================ PERIPHERALS ================
    ask_controller = make_branch(
        'Do you play with a controller?',
        standard_choices
    )
    if ask_controller() == 'yes':
        build['Controller'] = ('8Bitdo Ultimate 2',200.0)
    else:
        build['Controller'] = ('No controller',0.0)

    ask_keyboard = make_branch(
        'Do you do a lot of typing?',
        standard_choices
    )
    if ask_keyboard() == 'yes':
        build['Keyboard'] = ('Kaihl Brown Mechanical', 250.0)
    else:
        build['Keyboard'] = ('Basic Keyboard', 100.0)
    
    ask_camera = make_branch(
        'Do you stream or record yourself?',
        standard_choices
    )
    if ask_camera() == 'yes':
        build['Camera'] = ('USB 4K Camera', 300.0)
    else:
        build['Camera'] = ('No camera', 0.0)

    #================ USED PARTS ================
    ask_used = make_branch(
        'Willing to buy used parts?',
        standard_choices
    )
    used = ask_used() =='yes'

    #================ BRANCH: USED ================
    if used:
        ask_tier = make_branch(
            'Which is the budget tier?',
            ['low','mid','high']
        )
        tier = ask_tier()
        match tier:
            case 'low':
                ask_gpu = make_branch('Do you need GPU?', standard_choices)
                if ask_gpu() == 'yes':
                    build['CPU'] = ('Xeon 2680v4', 400.0)
                    build['RAM'] = ('16GB DDR4', 200.0)
                    build['GPU'] = ('RX 580', 500.0)
                else:
                    build['CPU'] = ('Xeon 2680v4', 400.0)
                    build['RAM'] = ('16GB DDR4', 200.0)
                    build['GPU'] = ('GT 750 Ti', 350.0)
            case 'mid':
                ask_gpu = make_branch('Do you need GPU?', standard_choices)
                if ask_gpu() == 'yes':
                    build['CPU'] = ('Ryzen 5700X3D', 800.0)
                    build['RAM'] = ('32GB DDR4', 350.0)
                    build['GPU'] = ('RX 6700', 1200.0)
                else:
                    build['CPU'] = ('Ryzen 5700X3D', 800.0)
                    build['RAM'] = ('32GB DDR4', 350.0)
                    build['GPU'] = ('RX 580', 500.0)
            case 'high':
                ask_gpu = make_branch('Do you need GPU?', standard_choices)
                if ask_gpu() == 'yes':
                    build['CPU'] = ('Ryzen 7800X3D', 1300)
                    build['RAM'] = ('64GB DDR4', 500)
                    build['GPU'] = ('RTX 3090', 3500)
                else:
                    build['CPU'] = ('Ryzen 7800X3D', 1300)
                    build['RAM'] = ('64GB DDR4', 500)
                    build['GPU'] = ('RX 6700', 1200)
        
    #================ BRANCH: NEW ================
    else:
        ask_tier = make_branch(
            'Which is the budget tier?',
            ['low','mid','high']
        )
        tier = ask_tier()
        match tier:
            case 'low':
                ask_gpu = make_branch('Do you need GPU?', standard_choices)
                if ask_gpu() == 'yes':
                    build['CPU'] = ('I5 12600F', 900.0)
                    build['RAM'] = ('16GB DDR4', 250.0)
                    build['GPU'] = ('RX 6600', 900.0)
                else:
                    build['CPU'] = ('Ryzen 5600G (APU)', 600.0)
                    build['RAM'] = ('16GB DDR4', 250.0)
                    build['GPU'] = ('Vega 8', 0.0)
            case 'mid':
                ask_gpu = make_branch('Do you need GPU?', standard_choices)
                if ask_gpu() == 'yes':
                    build['CPU'] = ('Ryzen 7600F', 1000.0)
                    build['RAM'] = ('32GB DDR5', 450.0)
                    build['GPU'] = ('RTX 4070', 2500.0)
                else:
                    build['CPU'] = ('Ryzen 7600F', 1000.0)
                    build['RAM'] = ('32GB DDR5', 450.0)
                    build['GPU'] = ('RX 6600', 900.0)
            case 'high':
                ask_gpu = make_branch('Do you need GPU?', standard_choices)
                if ask_gpu() == 'yes':
                    build['CPU'] = ('Ryzen 9800X3D', 2000.0)
                    build['RAM'] = ('64GB DDR5', 700.0)
                    build['GPU'] = ('RTX 4090', 5500.0)
                else:
                    build['CPU'] = ('Ryzen 9800X3D', 2000.0)
                    build['RAM'] = ('64GB DDR5', 700.0)
                    build['GPU'] = ('RTX 4070', 2500.0)

    #================ Power Supply Unit (Font) ================
    ask_upgrade = make_branch(
        'Do you intend on upgrading this PC in the future?',
        standard_choices
    )
    if ask_upgrade() == 'yes':
        build['PSU'] = ('700W 80+ Bronze', 300.0)
    else:
        build['PSU'] = ('500W 80+ Bronze',200.0)

    #================ MONITOR ================
    ask_entertainment = make_branch(
        'Do you intend on watching entertainment in this PC?',
        standard_choices
    )
    if ask_entertainment() == 'yes':
        build['Monitor'] = ('27" 1440p + Soundbar', 1200.0)
    else:
        build['Monitor'] = ('23" 1440p', 800.0)

    ask_competitive = make_branch(
        'Do you play competitive games?',
        standard_choices
    )
    if ask_competitive == 'yes':
        build['FPS'] = ('144 Hz', 200.0)
    else:
        build['FPS'] = ('60 Hz', 0.0)
    
    #================ WIFI ================
    ask_wifi = make_branch(
        'Do you have access to an ethernet cable for the PC?',
        standard_choices
    )
    if ask_wifi() == 'yes':
        build['Wifi'] = ('Bluetooth Dongle (Optional)', 50.0)
    else:
        build['Wifi'] = ('PCIe Wi-Fi AX200', 150.0)

    #================ FINAL PRINT ================
    print_build(build)

#================ EXECUTION ================
if __name__ == "__main__":
    main()