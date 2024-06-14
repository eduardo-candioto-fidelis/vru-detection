# VRU Detection
---

This repository contains code for simulating pedestrian (VRU) movements at crosswalks and methods for detecting them. The key features are:

- Simulation of pedestrians crossing streets using the `CARLA` simulator.
- Dataset creation from the simulations.
- Instructions for training a detection model with `OpenPCDet`.

## How to Run the Simulation
---

To run the simulation, follow these steps:

1. Install the `CARLA` simulator.
2. Copy the `run_simulation.py` file to the `PythonAPI/` folder inside the `CARLA` directory.
3. Execute the Python script with the desired arguments. For example:
    ```bash
    python run_simulation.py -f 300 -w 200 -n 0 --points-per-second 1300000 -R 20
    ```

## Training the Detection Model
---

To train a detection model, follow these steps:

1. Clone the `OpenPCDet` repository and install it as instructed:
    ```bash
    git clone https://github.com/open-mmlab/OpenPCDet.git
    ```
2. Follow the instructions in `CUSTOM_DATASET_TUTORIAL.md` using the dataset created with the simulator.
3. Proceed with the instructions in `GETTING_STARTED.md` to train and test the model.

## License

This project is licensed under the MIT License.