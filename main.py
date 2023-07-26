from src.Algorithm.DynamicProgramming.LineWorld import LineWorldDynamicProgramming
from src.Algorithm.DynamicProgramming.GridWorld import GridWorldDynamicProgramming
from src.Algorithm.DynamicProgramming.Secret1 import Secret1DynamicProgramming
from src.Algorithm.MonteCarlo.LineWorld import LineWorldMonteCarlo
from src.Algorithm.MonteCarlo.GridWorld import GridWorldMonteCarlo
from src.Algorithm.MonteCarlo.Secret1 import Secret1MonteCarlo


if __name__ == "__main__":
    print("Methods family - \033[1mDynamicProgramming\033[0m")
    print("\n")
    
    (LineWorldDynamicProgramming()).execute()
    print("\n")
    (GridWorldDynamicProgramming()).execute()
    print("\n")
    Secret1DynamicProgramming().execute()
    print("\n")

    print("Methods family - \033[1mMonteCarlo\033[0m")
    print("\n")

    (LineWorldMonteCarlo()).execute()
    print("\n")
    (GridWorldMonteCarlo()).execute()
    print("\n")
    # (Secret1MonteCarlo()).execute()
    # print("\n")
    
