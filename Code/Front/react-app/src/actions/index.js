export const SET_ACTIVE_COMPONENT = "SET_ACTIVE_COMPONENT";
export const SET_CANDIDATES = "SET_CANDIDATES";
export const SET_DATASET = "SET_DATASET";
export const TOGGLE_MASK = "TOGGLE_MASK";
export const PRUNE_POINTS = "PRUNE_POINTS";
export const SET_LEFT_POINTS = "SET_LEFT_POINTS";
export const SET_MODE = "SET_MODE";
export const RESTART = "RESTART";
export const UPDATE_CONVEX_HULL = "UPDATE_CONVEX_HULL";
export const SET_SCORE = "SET_SCORE";
export const SET_NAME = "SET_NAME";
export const SET_EMAIL = "SET_EMAIL";
export const SET_INTERACTION_RESULT = "SET_INTERACTION_RESULT";
export const RENAME_NODE = "RENAME_NODE";
export const SET_K = "SET_K";
export const SET_RADIUS = "SET_RADIUS"
export const SET_SELECTEDDATASET = "SET_SELECTEDDATASET"

export const SET_LABEL = "SET_LABEL"

export const setRadius = radius => ({
  type: SET_RADIUS,
  radius
});

export const setRenameNode = renameNode => ({
  type: RENAME_NODE,
  renameNode
});

export const setInteractionResult = interactionResults => ({
  type: SET_INTERACTION_RESULT,
  interactionResults
});

export const setUserName = name => ({
  type: SET_NAME,
  name
});

export const setUserEmail = email => ({
  type: SET_EMAIL,
  email
});

export const setActiveComponent = component => ({
  type: SET_ACTIVE_COMPONENT,
  component
});

export const setCandidates = candidates => ({
  type: SET_CANDIDATES,
  candidates
});

export const setDataset = (points, mask, attributes) => ({
  type: SET_DATASET,
  points,
  mask,
  attributes
});

export const toggleMask = (attr, val) => ({
  type: TOGGLE_MASK,
  mask: { [attr]: val }
});

export const prunePoints = (indices, step) => ({
  type: PRUNE_POINTS,
  indices,
  step
});

export const setLeftPoints = indices => ({
  type: SET_LEFT_POINTS,
  indices
});

export const setScore = scores => ({
  type: SET_SCORE,
  scores
})

export const changeMode = mode => ({
  type: SET_MODE,
  mode
});

export const changeK = K => ({
  type: SET_K,
  K
});

export const changeSelectedDataset = ds => ({
  type: SET_SELECTEDDATASET,
  ds
});

export const restart = () => ({
  type: RESTART
});

export const updateConvexHull = vertices => ({
  type: UPDATE_CONVEX_HULL,
  vertices
});
