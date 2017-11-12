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
        '''if not isinstance(observation_space, discrete.Discrete):
            raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        
        if not isinstance(action_space, discrete.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        '''
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = len(action_space)
        self.eta = 0.1
        self.eps = 0.30
        self.discount = 0.95
        self.maxIters = 10000
        self.numIters = 0
        self.weights = defaultdict(float)
        self.gameNumber=0

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
    def noNewState(self, state):
        return len(state)==0
    def incorporateFeedback(self, state, action, reward, newState=()):
        DeltaWeights=defaultdict(float)
        #for f,v in self.featureExtractor(state,action):
        #    self.weights[f]= 0 
        Prediction=self.getQ(state,action)
        eta=self.getStepSize()
        if len(newState)>0:
            V=[self.getQ(newState,newAction) for newAction in self.action_space]
            Vopt=max(V)
            Target=reward+self.discount*Vopt
        else:
            Target=reward
        DeltaScalar=self.eta*(Prediction-Target)
        for key,v in self.feature_extractor(state,action):
            DeltaWeights[key]= DeltaScalar * v 
        for key in self.weights.keys():
            self.weights[key]=self.weights.get(key,0)-DeltaWeights.get(key,0)
    # Define the feature extractor
    def feature_extractor(self, state, action):
        phi=[]
        featureValue = 1
        fk1 = (tuple(state), action)

        phi.append((fk1,featureValue))
        for i, v in enumerate(state):
            fk2=(i,v,action)
            phi.append((fk2,featureValue))
        #print('phi for iteration: %s new phi is: %s' % (self.numIters, phi))
        return phi

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.eps
        # epsilon greedy.
        if numpy.random.random() <= eps:
            action=numpy.random.choice(self.action_space)
        else:
            action = max((self.getQ(observation, action), action) for action in self.action_space)[1] 
        return action

    def learn(self, env):
        obs = env.reset()
        for t in range(self.maxIters):
            self.numIters += 1
            action = self.act(obs)
            if self.numIters % 300 ==0:
                print('GameNumber:"%s" Iter "%s", action: "%s"' % (self.gameNumber,self.numIters, action))
            obs2, reward, done, _ = env.step(action)
            #print ('Iter "%s", IsEnd: "%s", reward: "%s", newState: "%s"' % (self.numIters, done, reward, obs2))
            if done:
                self.gameNumber += 1
                self.incorporateFeedback(obs, action, reward)
                break
            self.incorporateFeedback(obs, action, reward, obs2)
            obs = obs2