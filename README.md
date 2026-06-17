# SSBM-RL

This project is still in its early stages, so there isn't much to read yet.

I am using the Dolphin Emulator to run the game on my PC.

## Build dependencies

- `libmelee` (pip install melee)

Huge thanks to the team behind libmelee. It saved a lot of time and allowed me to skip straight to the fun part: building a Smash-playing AI.

---

## Current agenda

- Expand the AI action space to allow a wider range of moves
- Rework the reward system to encourage better gameplay behavior
- Build a tool to manage multiple training environments in parallel to speed up training
- Fix the currently broken feature that assigns opponents with random difficulty levels
- Run a deep training session to evaluate the current pipeline and refine training parameters
- Increase the skill ceiling by letting two AIs play against each other