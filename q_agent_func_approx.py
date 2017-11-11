from collections import defaultdict
import numpy
from gym.spaces import discrete

class UnsupportedSpace(Exception):
    pass

class QAgentFuncApprox(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, discrete.Discrete):
            raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        if not isinstance(action_space, discrete.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "eta" : 0.1,
            "eps": 0.05,            # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 10000}        # Number of iterations
        self.config.update(userconfig)
        self.weights = defaultdict(float)

        # TODO define Q_opt as self.weights dotproduct phi(s,a) (similar to get_Qopt in blackjack 4a assignment)
        self.q = defaultdict(lambda: self.config["init_std"] * numpy.random.randn(self.action_n) + self.config["init_mean"])
    def getQ(self, state, action):
        score = 0
        for f, v in self.feature_extractor(state, action):
            score += self.weights[f] * v
        return score



    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / numpy.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        DeltaWeights=defaultdict(float)
        #for f,v in self.featureExtractor(state,action):
        #    self.weights[f]= 0 
        Prediction=self.getQ(state,action)
        eta=self.getStepSize()
        if newState!=None:
            V=[self.getQ(newState,newAction) for newAction in self.action_space]
            Vopt=max(V)
            Target=reward+self.config["discount"]*Vopt
        else:
            Target=reward
        DeltaScalar=self.config["eta"]*(Prediction-Target)
        for key,v in self.feature_extractor(state,action).items():
            DeltaWeights[key]= DeltaScalar * v 
        for key in self.weights.keys():
            self.weights[key]=self.weights.get(key,0)-DeltaWeights.get(key,0)
    # Define the feature extractor
    # TODO include action in feature vector
    def feature_extractor(self, state, action):
        phi = defaultdict(float)
        phi[(state,action)] = 1
        return phi

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        # TODO update the argmax argument based on the new Q_opt function approx definition
        if numpy.random.random() <= eps:
            action=self.action_space.sample()
        else:
            action = max((self.getQ(observation, action), action) for action in self.action_space)[1] 
        return action

    def learn(self, env):
        config = self.config
        obs = env.reset()
        q = self.q
        for t in range(config["n_iter"]):
            action, _ = self.act(obs)
            obs2, reward, done, _ = env.step(action)
            if done:
                break
            self.incorporateFeedback(obs, action, reward, obs2)
            obs = obs2