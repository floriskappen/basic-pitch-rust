/* PORTED NUMPY FUNCTIONS */

/// Returns the location of the maximum element in the array.
///
/// # Arguments
///
/// * `arr` - Input array.
///
/// # Returns
///
/// * The location of the maximum element in the array, or `None` if the array is empty.
pub fn arg_max(arr: &[f32]) -> Option<usize> {
    if arr.is_empty() {
        None
    } else {
        Some(arr.iter().enumerate().reduce(|(max_index, max_value), (current_index, current_value)| {
            if current_value > max_value {
                (current_index, current_value)
            } else {
                (max_index, max_value)
            }
        }).unwrap().0)
    }
}

/// Returns the location of the maximum element in each row.
///
/// # Arguments
///
/// * `arr` - Input 2D array.
///
/// # Returns
///
/// * A vector containing the location of the maximum element in each row.
pub fn arg_max_axis1(arr: &[Vec<f32>]) -> Vec<Option<usize>> {
    arr.iter().map(|row| arg_max(row)).collect()
}

/// Returns the locations of elements in a 2D array that are greater than a given threshold.
///
/// # Arguments
///
/// * `arr2d` - The input 2D array.
/// * `threshold` - The value below which we want to filter out.
///
/// # Returns
///
/// * A pair of vectors with the first representing axis 0 and the second representing axis 1. 
///   These vectors contain the locations of `arr2d` which have values greater than the threshold.
pub fn where_greater_than_axis1(arr2d: &[Vec<f32>], threshold: f32) -> (Vec<usize>, Vec<usize>) {
    let mut output_x = Vec::new();
    let mut output_y = Vec::new();

    for (i, row) in arr2d.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value > threshold {
                output_x.push(i);
                output_y.push(j);
            }
        }
    }

    (output_x, output_y)
}

/// Calculate mean and standard deviation for a 2D-array.
///
/// # Arguments
///
/// * `array` - Array to find mean and standard deviation for.
///
/// # Returns
///
/// * A tuple with the mean and standard deviation.
pub fn mean_std_dev(array: &[Vec<f32>]) -> (f32, f32) {
    let (sum, sum_squared, count) = array.iter().fold((0.0, 0.0, 0), |prev, row| {
        let (row_sum, row_sums_squared, row_count) = row.iter().fold((0.0, 0.0, 0), |p, &value| {
            (p.0 + value, p.1 + value * value, p.2 + 1)
        });
        (prev.0 + row_sum, prev.1 + row_sums_squared, prev.2 + row_count)
    });

    let mean = sum / count as f32;
    let std_dev = ((1.0 / (count as f32 - 1.0)) * (sum_squared - (sum * sum) / count as f32)).sqrt();
    (mean, std_dev)
}

/// Calculate the global max value in a 2D array. This is equivalent to numpy.max.
///
/// # Arguments
///
/// * `array` - Array to calculate max over.
///
/// # Returns
///
/// * The maximum value in the array.
pub fn global_max(array: &[Vec<f32>]) -> f32 {
    array.iter().fold(0.0, |prev, row| prev.max(*row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()))
}

/// Calculate the minimum over axis 0 for a 3D array.
///
/// # Arguments
///
/// * `array` - Array to calculate min over.
///
/// # Returns
///
/// * A 2D array where each element represents the minimum for a fixed first dimension.
pub fn min_3d_for_axis0(array: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
    let mut min_array = array[0].clone();
    
    for x in 1..array.len() {
        for y in 0..array[0].len() {
            for z in 0..array[0][0].len() {
                min_array[y][z] = min_array[y][z].min(array[x][y][z]);
            }
        }
    }

    min_array
}

/// Calculate the relative extrema in an array over axis 0 assuming clipped edges.
/// A Rust implementation of scipy.signal.argrelmax
/// https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html
///
/// Relative extrema are calculated by finding locations where data[n] > data[n+1:n+order+1]
/// is true.
///
/// # Arguments
///
/// * `array` - Array to find the relative maxima.
/// * `order` - How many points on each side to use for the comparison to consider comparator(n, n+x)
///   to be true.
///
/// # Returns
///
/// * Indices of the maxima. Each element represents indices of the location in data.
///   This does not match scipy which returns an n-d tuple with each dimension representing an axis of the data.
pub fn arg_rel_max(array: &[Vec<f32>], order: usize) -> Vec<(usize, usize)> {
    let mut result = Vec::new();

    for col in 0..array[0].len() {
        for row in 0..array.len() {
            let mut is_rel_max = true;

            for comparison_row in row.saturating_sub(order)..=usize::min(array.len() - 1, row + order) {
                if comparison_row != row && array[row][col] <= array[comparison_row][col] {
                    is_rel_max = false;
                    break;
                }
            }

            if is_rel_max {
                result.push((row, col));
            }
        }
    }

    result
}

/// Calculate the maximum over axis 0 for a 3D array.
///
/// # Arguments
///
/// * `array` - Array to calculate max over.
///
/// # Returns
///
/// * A 2D array where each element represents the maximum for a fixed first dimension.
pub fn max_3d_for_axis0(array: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
    let mut max_array = array[0].clone();

    for x in 1..array.len() {
        for y in 0..array[0].len() {
            for z in 0..array[0][0].len() {
                max_array[y][z] = max_array[y][z].max(array[x][y][z]);
            }
        }
    }

    max_array
}