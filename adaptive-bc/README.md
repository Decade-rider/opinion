# adaptive-bc

repository for adaptive bounded confidence code

## system:

Apple M3

### notes about runs:

can take around 5 - 10 minutes to run

i have 32 GB of memory and the program ate it all for K = 1,
suggesting to increase K so model can converge faster
(K is how many agents update their opinion in each time step)
baseline adaptive-bc managed to run, model did not converge as we can see in picture

### how to run:

can adjust params if needed

`python run.py` to run simulation and save data

`python visualize.py` to see opinion evolution from saved data
