import melee
import argparse
import signal
import sys
import gymnasium as gym


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Debug mode. Creates a CSV of all game states')
    parser.add_argument('--address', '-a', default="127.0.0.1",
                        help='IP address of Slippi/Wii')
    parser.add_argument('--dolphin_executable_path', '-e', default=None,
                        help='The directory where dolphin is')
    parser.add_argument('--iso', default=None, type=str,
                        help='Path to melee iso.')
    args = parser.parse_args()

    
    log = None
    if args.debug:
        log = melee.Logger()


    console = melee.Console(
        path=args.dolphin_executable_path,
        slippi_address=args.address,
        logger=log,
        save_replays=args.debug,
        fullscreen=False,
    )

    ports = [1, 2]

    controllers = {
        port: melee.Controller(
            console=console,
            port=port,
            type=melee.ControllerType.STANDARD)
        for port in ports
    }

    # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
    def signal_handler(sig, frame):
        for controller in controllers.values():
            controller.disconnect()
        console.stop()
        if args.debug:
            log.writelog()
            print("") #because the ^C will be on the terminal
            print("Log file created: " + log.filename)
        print("Shutting down cleanly...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    
    console.run(iso_path=args.iso)
    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    print("Console connected")

    print("Connecting controller to console...")

    for controller in controllers.values():
        if not controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
    print("Controller connected")

    custome = 0
    framedata = melee.framedata.FrameData()
    menu_helper = melee.MenuHelper()


    class MeleeEnv(gym.Env):
        def reset(self): # Does a soft reset of the game, by sending inputs to get through the menu, and then starting a new game.
            gamestate = console.step() # Step the console forward one frame, and receive the new gamestate
            if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                while gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                    for port, controller in controllers.items():
                        menu_helper.menu_helper_simple(
                        gamestate,
                        controller,
                        melee.Character.FOX,
                        melee.Stage.YOSHIS_STORY,
                        costume=port,
                        autostart=port == 1,
                        swag=False)
                    if log:
                        log.skipframe()
                    gamestate = console.step() # Step the console forward one frame, and receive the new gamestate
            if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                while gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                    for port, controller in controllers.items():
                        menu_helper.menu_helper_simple(
                        gamestate,
                        controller,
                        melee.Character.FOX,
                        melee.Stage.YOSHIS_STORY,
                        costume=port,
                        autostart=port == 1,
                        swag=False)
                    if log:
                        log.skipframe()
                    gamestate = console.step() # Step the console forward one frame, and receive the new gamestate

        def step(self, action):


    #MAIN LOOP
    while True:
        gamestate = console.step() # Step the console forward one frame, and receive the new gamestate
        if gamestate is None: # If no Info is received, skip the rest of the loop and try again on next frame
            continue

        if console.processingtime * 1000 > 12:
            print("WARNING: Last frame took " + str(console.processingtime*1000) + "ms to process.")

        # What menu are we in?
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            for port, controller in controllers.items():
                # THIS IS THE MAIN GAME LOOP
                melee.techskill.multishine(ai_state=gamestate.players[port], controller=controller)

            # Log info if in game
            if log:
                log.logframe(gamestate)
                log.writeframe()
        else:
            for port, controller in controllers.items():
                menu_helper.menu_helper_simple(
                    gamestate,
                    controller,
                    melee.Character.FOX,
                    melee.Stage.YOSHIS_STORY,
                    costume=port,
                    autostart=port == 1,
                    swag=False)
            if log:
                log.skipframe()