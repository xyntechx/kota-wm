Full env is fully observable 2D grid of roads

Task is to follow directions while abiding by traffic lights (ghost blocks on roads -- can be intercepted, but negative reward if intercepted)

Instructions are like:
```
1. Turn left in 5 units
2. Turn right in 2 units
3. Turn left on 4th avenue
```
note: "turn left in 2 units" requires memory, "turn left on 1st avenue" doesn't

Input is thus current instruction and current state of grid

Task tokens given only at the time step that task is given (not repeated so that we have have reference as to when the task was given so we know, for eg, "turn left in 5 blocks" from which starting block?)

Rewards:
✅ +20 at the instant where instruction must be followed and instruction is followed

✅ -20 at the instant where instruction must be followed but instruction is NOT followed

✅ -5 if stopping at a traffic light intersection

✅ +1 for every step where player follows traffic flow (not driving in the opposite lane)

✅ -5 for every step where player does NOT follow traffic flow (is driving in the opposite lane, tries U-turning)

✅ +1 for every step where player no-ops in front of an active red light (traffic light on for 5 ticks)

✅ -5 when player cuts active red light

✅ -20 when player (attempts to) exits the grid / go into a building

✅ 0 if player no-ops

✅ -5 if player no-ops more than or equal to 5 times in a row