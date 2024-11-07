// read points from a text file
function parsePoints(text) {
  const points = text
    .trim()
    .split("\n")
    .slice(1) // the first line is <numOfPoints> <dimension>
    .map(line =>
      line
        .trim()
        .split(/\s+/)
        .map(str => parseFloat(str))
    );
  return points; //obtain a 2D array
}

// read labels from a text file
function parseLabels(text) {
  const labels = text.trim().split("\n");
  return labels;
}

// check if val is in the range [low, high].
function isInRange(val, low, high) {
  return low <= val && val <= high;
}

// get the range of each attribute in a set of points.
export const getRanges = points => {
  const ranges = points[0].map(x => ({ low: x, high: x }));
  points.slice(1).forEach(point => {
    point.forEach((x, i) => {
      ranges[i].low = Math.min(ranges[i].low, x);
      ranges[i].high = Math.max(ranges[i].high, x);
    });
  });
  return ranges; //obtain an array of objects
};

// load a dataset by reading its points and labels.
export const loadCarDataset = async (pointsURL) => {
  let response = await fetch(pointsURL);
  let text = await response.text();
  const cars = text
      .trim()
      .split("\n")
      .map(line => {
        const parts = line.trim().split(/\s+/);
        const strings = parts.slice(0, 3);  // the first three
        const numbers = parts.slice(3).map(part => +part); // the rest
        return [...strings, ...numbers];
      });
  return cars;
};


// load a dataset by reading its categorical attributes
export const loadDataset = async (pointsURL, labelsURL) => {
  let response = await fetch(pointsURL);
  const points = parsePoints(await response.text());
  if (labelsURL === undefined) {
    return points;
  }
  response = await fetch(labelsURL);
  const labels = parseLabels(await response.text());
  return [points, labels];
};

// get points that are in the specified ranges.
export const selectCandidates = (points, ranges, mask, maxPoints) => {
  const candidates = [];
  for (let i = 0; i < points.length; ++i)
  {
    if (candidates.length >= maxPoints) break;
    const point = [];
    //categorical
    for (let j = 0; j < 3; ++j)
    {
      if (mask[j])
        point.push(points[i][j]);
    }
    //numerical
    let isValid = true;
    for (let j = 3; j < points[0].length; ++j)
    {
      if (mask[j] && !isInRange(points[i][j], ranges[j - 3][0], ranges[j - 3][1])) {
        isValid = false;
        break;
      }
      else if(mask[j])
        point.push(points[i][j]);
    }

    if (isValid) {
      candidates.push(point);
    }
  }
  return candidates
};

function findMaxMinOfColumns(matrix) {
  let maxValues = [];
  let minValues = [];

  if (matrix.length === 0) return [minValues, maxValues];

  for (let col = 0; col < matrix[0].length; col++)
  {
    let maxInCol = matrix[0][col];
    let minInCol = matrix[0][col];

    for (let row = 1; row < matrix.length; row++)
    {
        if (matrix[row][col] > maxInCol) {
          maxInCol = matrix[row][col];
        }
        if (matrix[row][col] < minInCol) {
          minInCol = matrix[row][col];
        }
    }

    maxValues.push(maxInCol);
    minValues.push(minInCol);
  }

  return [minValues, maxValues];
}

export const normalized = (points, smallerBetter) => {
  const values = findMaxMinOfColumns(points);
  console.log(values);
  const candidates = [];
  for (let i = 0; i < points.length; ++i)
  {
    const point = [];
    //categorical
    for (let j = 0; j < points[0].length - smallerBetter.length; ++j)
        point.push(points[i][j]);
    //numerical
    for (let j = points[0].length - smallerBetter.length; j < points[0].length; ++j)
    {
        if(smallerBetter)
          point.push(1 - (points[i][j] - values[0][j]) / (values[1][j] - values[0][j]));
        else
          point.push((points[i][j] - values[0][j]) / (values[1][j] - values[0][j]));
    }
      candidates.push(point);
  }
  return candidates
};


// get points that are in the specified ranges.
export const selectCatonlyCandidates = (points, mask) => {
  const candidates = [];
  const seen = new Set();

  for (let i = 0; i < points.length; ++i)
  {
    const tuple = [];
    for (let j = 0; j < 3; ++j)
      if (mask[j]) tuple.push(points[i][j]);

    const tupleStr = JSON.stringify(tuple); // 将数组转换为字符串
    if (!seen.has(tupleStr)) { // 如果这是一个新的子数组
      candidates.push(tuple);
      seen.add(tupleStr); // 添加到已见过的集合中
    }
  }
  return candidates;
};


// convert a JS array to a C++ 2D vector
export const array2Vector2D = array => {
  const vector = new window.Module.VecVecDouble();
  array.forEach(arr => {
    const vec = new window.Module.VectorDouble();
    arr.forEach(x => vec.push_back(x));
    vector.push_back(vec);
    vec.delete();
  });
  return vector;
};

// convert a C++ 2D vector to a JS array
export const vector2Array2D = vector => {
  const array = [];
  for (let i = 0; i < vector.size(); ++i) {
    const vec = vector.get(i);
    const arr = [];
    for (let j = 0; j < vec.size(); ++j) arr.push(vec.get(j));
    array.push(arr);
  }
  return array;
};

// convert a C++ vector to a JS array
export const vector2Array = vector => {
  const array = [];
  for (let i = 0; i < vector.size(); ++i) {
    array.push(vector.get(i));
  }
  return array;
};

// get the indices of points pruned.
// both prevIndices and currIndices need to be sorted.
export const getPrunedIndices = (prevIndices, currIndices) => {
  let prunedIndices = [];
  for (let i = 0, j = 0; i < prevIndices.size() || j < currIndices.size(); ) {
    if (j >= currIndices.size() || prevIndices.get(i) < currIndices.get(j)) {
      prunedIndices.push(prevIndices.get(i));
      ++i;
    } else {
      ++i;
      ++j;
    }
  }
  return prunedIndices;
};


const dominates = (p1, p2, smallerBetter, isSelected) => {
  //categorical
  for (let i = 0; i < 3; i++)
  {
      if(p1[i] !== p2[i])
        return 0;
  }

  //numerical
  for (let i = 3; i < p1.length; i++)
  {
    if (isSelected.at(i) === 1)
    {
      if (smallerBetter[i] === 1) {
        if (p1[i] > p2[i]) return 0;
      } else {
        if (p1[i] < p2[i]) return 0;
      }
    }
  }
  return 1;
}


export const getSkyline = (points, smallerBetter, isSelected) => {
  let i, j, dominated, index = 0, m;
  console.log(smallerBetter, isSelected);
  let sl = new Array(points.length);
  for (i = 0; i < points.length; i++)
  {

    dominated = 0;
    const pt = points.at(i).slice();

    for (j = 0; j < index && !dominated; ++j) {
      if (dominates(points[sl[j]], pt, smallerBetter, isSelected))
        dominated = 1;
    }

    if (!dominated)
    {
      m = index;
      index = 0;
      for (j = 0; j < m; j++) {
        if (!dominates(pt, points[sl[j]], smallerBetter, isSelected)) {
          sl[index++] = sl[j];
        }
      }
      sl[index++] = i;
    }
  }

  console.log(sl);

  let skyline = [];
  for(let i = 0; i < index; ++i)
    skyline.push(points[sl[i]]);

  return skyline;
}