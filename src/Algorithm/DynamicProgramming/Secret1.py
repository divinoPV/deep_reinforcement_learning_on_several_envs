from src.do_not_touch.result_structures import (
    ValueFunction,
    PolicyAndValueFunction,
)
from src.env.Secret1 import Secret1
from src.Algorithm.DynamicProgramming import DynamicProgramming


class Secret1DynamicProgramming(Secret1):
    def __init__(self):
        super().__init__()

        self.mdp = DynamicProgramming()
    
    def policy_evaluation(self) -> ValueFunction:
        """ Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy.
            Returns the Value function (V(s)) of this policy.
        """

        return self.mdp.policy_evaluation(
            self.states,
            self.actions,
            self.rewards,
            self.transition_probability,
            self.random_probability,
        )
    
    def policy_iteration(self) -> PolicyAndValueFunction:
        """ Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function.
            Returns the Policy (Pi(s,a)) and its Value Function (V(s)).
        """

        return self.mdp.policy_iteration(
            self.states,
            self.actions,
            self.rewards,
            self.transition_probability,
        )
    
    def value_iteration(self) -> PolicyAndValueFunction:
        """ Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function.
            Prints the Policy (Pi(s,a)) and its Value Function (V(s)).
        """
        
        return self.mdp.value_iteration(
            self.states,
            self.actions,
            self.rewards,
            self.transition_probability,
        )

    def execute(self):
        print(f"Environment \033[1m{self.__class__.__name__}\033[0m")
        print("\n")
        
        print("\t \033[1mPolicy Evaluation\033[0m")
        print("\t", self.policy_evaluation())
        print("\n")
        
        print("\t \033[1mPolicy Iteration\033[0m")
        print("\t", self.policy_iteration())
        print("\n")
        
        print("\t \033[1mValue Iteration\033[0m")
        print("\t", self.value_iteration())
        print("\n")
        
        print("-------------------------------------------------------------------------------------------------------")
