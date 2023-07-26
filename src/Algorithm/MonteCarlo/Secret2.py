from src.do_not_touch.result_structures import PolicyAndActionValueFunction
from src.env.Secret2 import Secret2


class Secret1MonteCarlo(Secret2):
    def __init__(self):
        super().__init__()
    
    def monte_carlo_es(self, num_episodes=1000, gamma=1.0) -> PolicyAndActionValueFunction:
        pass
    
    def on_policy_first_visit_monte_carlo_control(self) -> PolicyAndActionValueFunction:
        pass
    
    def off_policy_monte_carlo_control(self) -> PolicyAndActionValueFunction:
        pass
    
    def execute(self):
        print(f"Environment \033[1m{self.__class__.__name__}\033[0m")
        print("\n")
        
        print("\t \033[1mMonte Carlo ES\033[0m")
        print("\t", self.monte_carlo_es())
        print("\n")
        
        print("\t \033[1mOn Policy First Visit Monte Carlo Control\033[0m")
        print("\t", self.on_policy_first_visit_monte_carlo_control())
        print("\n")
        
        print("\t \033[1mOff Policy Monte Carlo Control\033[0m")
        print("\t", self.off_policy_monte_carlo_control())
        print("\n")
