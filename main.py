import melee
import argparse
import signal
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Debug mode. Creates a CSV of all game states",
    )
    parser.add_argument("--address", "-a", default="127.0.0.1", help="IP address of Slippi/Wii")
    parser.add_argument(
        "--dolphin_executable_path",
        "-e",
        default=None,
        help="The directory where dolphin is",
    )
    parser.add_argument("--iso", default=None, type=str, help="Path to melee iso.")
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

    controllers = {port: melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD) for port in ports}

    # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
    def signal_handler(sig, frame):
        for controller in controllers.values():
            controller.disconnect()
        console.stop()
        if args.debug:
            log.writelog()
            print("")  # because the ^C will be on the terminal
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

    def get_observation(gamestate, ai_port, opponent_port):
        ai = gamestate.players[ai_port]
        opp = gamestate.players[opponent_port]

        obs = np.array(
            [
                ai.position.x / 100.0,
                ai.position.y / 100.0,
                opp.position.x / 100.0,
                opp.position.y / 100.0,
                float(ai.facing) * 2 - 1,
                float(opp.facing) * 2 - 1,
                ai.action.value / 400.0,
                opp.action.value / 400.0,
                ai.action_frame / 60.0,
                opp.action_frame / 60.0,
                ai.jumps_left / 2.0,
                opp.jumps_left / 2.0,
                ai.stock / 4.0,
                opp.stock / 4.0,
                ai.percent / 300.0,
                opp.percent / 300.0,
                ai.off_stage,
                opp.off_stage,
                float(ai.is_powershield),
                float(opp.is_powershield),
                gamestate.distance / 100,
            ],
            dtype=np.float32,
        )
        return obs

    class MeleeEnv(gym.Env):
        def __init__(self, console, controllers, menu_helper, ai_port, opponent_port, log=None, opponent_cpu_level=2, action_repeat=4):
            super().__init__()
            self.console = console
            self.controllers = controllers
            self.menu_helper = menu_helper
            self.log = log
            self.ai_port = ai_port
            self.opponent_port = opponent_port
            self.opponent_cpu_level = opponent_cpu_level
            self.action_repeat = action_repeat
            self.previous_gamestate = None

            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

            self.action_space = gym.spaces.Discrete(7)  # n = number of possible actions

        def do_action(self, action, controller):
            controller.release_all()
            # Default: neutral stick. Specific actions below override this.
            controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.5, 0.5)

            controller.press_shoulder(melee.Button.BUTTON_L, 0.0)

            if action == 0 or action == 4:  # no action or temp shield
                pass
            elif action == 1:  # move left
                controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.0, 0.5)
            elif action == 2:  # move right
                controller.tilt_analog(melee.Button.BUTTON_MAIN, 1.0, 0.5)
            elif action == 3:  # jump
                controller.press_button(melee.Button.BUTTON_X)
            elif action == 4:  # shield
                controller.press_shoulder(melee.Button.BUTTON_L, 1.0)
            elif action == 5:  # attack
                controller.press_button(melee.Button.BUTTON_A)
            elif action == 6:  # special
                controller.press_button(melee.Button.BUTTON_B)

        def _compute_reward(self, gamestate):
            if self.previous_gamestate is None:
                self.previous_gamestate = gamestate
                return 0.0
            ai_now = gamestate.players[self.ai_port]
            opp_now = gamestate.players[self.opponent_port]
            ai_prev = self.previous_gamestate.players[self.ai_port]
            opp_prev = self.previous_gamestate.players[self.opponent_port]

            reward = 0.0

            # Reward for damage dealt
            reward += (opp_now.percent - opp_prev.percent) * 0.01

            # Damage taken by AI is negative reward
            reward -= (ai_now.percent - ai_prev.percent) * 0.01

            # Stock lost by opponent is positive reward
            if opp_now.stock < opp_prev.stock:
                reward += 1.0

            # Stock lost by AI is negative reward
            if ai_now.stock < ai_prev.stock:
                reward -= 1.0
            self.previous_gamestate = gamestate
            return reward

        def _navigate_to_game(self, gamestate):
            """Step through menus (including postgame) until a match is running."""

            if self.opponent_cpu_level == 10:
                cpu_lvl = random.randint(1, 9)
            else:
                cpu_lvl = self.opponent_cpu_level
            while gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                if gamestate.menu_state == melee.Menu.POSTGAME_SCORES:
                    # Spam start to get past the results screen
                    self.menu_helper.skip_postgame(self.controllers[self.ai_port])
                else:
                    # Iterate through controllers and navigate menus for each port
                    for port, controller in self.controllers.items():
                        if port == self.ai_port:
                            self.menu_helper.menu_helper_simple(
                                gamestate,
                                controller,
                                melee.Character.LUIGI,
                                melee.Stage.YOSHIS_STORY,
                                costume=port,
                                autostart=False,
                                swag=False,
                                cpu_level=0,
                            )
                        elif port == self.opponent_port:
                            self.menu_helper.menu_helper_simple(
                                gamestate,
                                controller,
                                melee.Character.MARIO,
                                melee.Stage.YOSHIS_STORY,
                                costume=port,
                                autostart=True,
                                swag=False,
                                cpu_level=cpu_lvl,
                            )

                if self.log:
                    self.log.skipframe()
                gamestate = self.console.step()
                while gamestate is None:
                    gamestate = self.console.step()
            return gamestate

        def reset(
            self, seed=None, options=None
        ):  # Does a soft reset of the game, by sending inputs to get through the menu, and then starting a new game.
            super().reset(seed=seed)
            gamestate = self.console.step()  # Step the console forward one frame, and receive the new gamestate

            while gamestate is None:
                gamestate = self.console.step()

            # If we're still mid-match (shouldn't normally happen, but just in case),
            # step forward until the match ends and we land on a menu screen.
            while gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                gamestate = self.console.step()
                while gamestate is None:
                    gamestate = self.console.step()

            # Now navigate menus (including postgame results) into a fresh match
            gamestate = self._navigate_to_game(gamestate)

            self.previous_gamestate = None

            # Sanity check: confirm opponent is actually set as CPU
            opp_cpu_level = gamestate.players[self.opponent_port].cpu_level
            print(f"Opponent CPU level reported by game: {opp_cpu_level}")

            return (get_observation(gamestate, self.ai_port, self.opponent_port), {"frame": gamestate.frame})

        def step(self, action):
            total_reward = 0.0
            terminated = False
            info = {"frame": 0}

            for _ in range(self.action_repeat):
                gamestate = self.console.step()

                while gamestate is None:
                    gamestate = self.console.step()

                if self.console.processingtime * 1000 > 12:
                    print("WARNING: Last frame took " + str(self.console.processingtime * 1000) + "ms")

                # apply same action for multiple frames
                self.do_action(action, self.controllers[self.ai_port])

                obs = get_observation(gamestate, self.ai_port, self.opponent_port)
                reward = self._compute_reward(gamestate)

                total_reward += reward
                info = {"frame": gamestate.frame}

                if self.log:
                    self.log.logframe(gamestate)
                    self.log.writeframe()

                terminated = gamestate.players[self.ai_port].stock == 0 or gamestate.players[self.opponent_port].stock == 0

                if terminated:
                    break

            return obs, total_reward, terminated, False, info

    env = MeleeEnv(
        console=console,
        controllers=controllers,
        menu_helper=menu_helper,
        ai_port=1,
        opponent_port=2,
        log=log,
        opponent_cpu_level=10,  # set low for training
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_0000)
    model.save("ppo_melee")


"""     obs, info = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward, terminated)
        if terminated:
            obs, info = env.reset() """
