#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <bitset>
#include <random>
#include <vector>
#include <algorithm>

// Convert int8 to binary string
std::string int8_to_bin(int8_t value) {
    std::bitset<8> bits(value);
    return bits.to_string();
}

// Convert binary string to int8
int8_t bin_to_int8(const std::string& binary) {
    std::bitset<8> bits(binary);
    return static_cast<int8_t>(bits.to_ulong());
}

// Convert float16 (represented as uint16_t) to binary string
std::string float16_to_bin(uint16_t value) {
    std::bitset<16> bits(value);  // Use 16 bits as float16 is represented by uint16_t
    return bits.to_string();
}

// Convert binary string to float16 (represented as uint16_t)
uint16_t bin_to_float16(const std::string& binary) {
    std::bitset<16> bits(binary);
    return static_cast<uint16_t>(bits.to_ulong());
}

// Random Modify method for int8 and float16
template<typename T>
void modify_bits(T* data, int num_elements, const std::vector<int>& bit_indices, const std::vector<double>& probabilities, int strategy) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_elements - 1);
    std::uniform_real_distribution<> prob(0.0, 1.0);

    for (size_t i = 0; i < bit_indices.size(); ++i) {
        int bit_index = bit_indices[i];
        double target_prob = probabilities[i];

        int num_modify = int((strategy == 1 ? 2 : 4) * std::abs(target_prob - 0.5) * num_elements);

        for (int j = 0; j < num_modify; ++j) {
            int index = dis(gen);
            std::string binary_rep = (sizeof(T) == sizeof(int8_t)) ? int8_to_bin(data[index]) : float16_to_bin(data[index]);
            
            if (target_prob > 0.5) {
                if (strategy == 2 && prob(gen) >= 0.75) {
                    binary_rep[bit_index] = '0';  // 25% chance to set bit to 0
                } else {
                    binary_rep[bit_index] = '1';  // Set bit to 1
                }
            } else {
                if (strategy == 2 && prob(gen) >= 0.75) {
                    binary_rep[bit_index] = '1';  // 25% chance to set bit to 1
                } else {
                    binary_rep[bit_index] = '0';  // Set bit to 0
                }
            }

            data[index] = (sizeof(T) == sizeof(int8_t)) ? bin_to_int8(binary_rep) : bin_to_float16(binary_rep);
        }
    }
}

// 修改后的 Silent Modify 方法
template<typename T>
void silent_modify_bits(T* data, int num_elements, const std::vector<int>& bit_indices, const std::vector<double>& probabilities, int strategy, double top_percent) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_elements - 1);
    std::uniform_real_distribution<> prob(0.0, 1.0);

    // 计算跳过的数量
    int num_skip = static_cast<int>(num_elements * top_percent);

    // 创建绝对值和索引对的向量
    std::vector<std::pair<double, int>> abs_values(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        abs_values[i] = std::make_pair(std::abs(data[i]), i);
    }

    // 按照绝对值从大到小排序
    std::sort(abs_values.begin(), abs_values.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first > b.first;
    });

    // 标记需要跳过的元素
    std::vector<bool> skip(num_elements, false);
    for (int i = 0; i < num_skip; ++i) {
        skip[abs_values[i].second] = true;
    }

    for (size_t i = 0; i < bit_indices.size(); ++i) {
        int bit_index = bit_indices[i];
        double target_prob = probabilities[i];

        int num_modify = int((strategy == 3 ? 2 : 4) * std::abs(target_prob - 0.5) * num_elements);
        int num_modified = 0;

        for (int j = 0; j < num_modify; ++j) {
            int index = dis(gen);

            // 跳过被标记的元素
            if (skip[index]) {
                // 尝试从其他元素中寻找可调整的权重
                if (num_modified < num_modify) {
                    continue;  // 继续查找非跳过的权重
                } else {
                    // 所有非 salient weight 都跳过时，对 salient weight 进行调整
                    break;
                }
            }

            std::string binary_rep = (sizeof(T) == sizeof(int8_t)) ? int8_to_bin(data[index]) : float16_to_bin(data[index]);

            if (target_prob > 0.5) {
                if (strategy == 4 && prob(gen) >= 0.75) {
                    binary_rep[bit_index] = '0';  // 25% 概率设为 0
                } else {
                    binary_rep[bit_index] = '1';  // 否则设为 1
                }
            } else {
                if (strategy == 4 && prob(gen) >= 0.75) {
                    binary_rep[bit_index] = '1';  // 25% 概率设为 1
                } else {
                    binary_rep[bit_index] = '0';  // 否则设为 0
                }
            }

            // 更新权重
            data[index] = (sizeof(T) == sizeof(int8_t)) ? bin_to_int8(binary_rep) : bin_to_float16(binary_rep);
            num_modified++;
        }

        // 如果跳过了所有元素，仍然没有达到调整目标，开始调整 salient weight
        if (num_modified < num_modify) {
            for (int j = 0; j < num_modify - num_modified; ++j) {
                int index = abs_values[j].second;  // 调整 salient weight
                std::string binary_rep = (sizeof(T) == sizeof(int8_t)) ? int8_to_bin(data[index]) : float16_to_bin(data[index]);

                // 重复调整逻辑
                if (target_prob > 0.5) {
                    binary_rep[bit_index] = '1';
                } else {
                    binary_rep[bit_index] = '0';
                }

                data[index] = (sizeof(T) == sizeof(int8_t)) ? bin_to_int8(binary_rep) : bin_to_float16(binary_rep);
            }
        }
    }
}


// Wrapper function for Python (int8 Random Modify)
static PyObject* modify_bits_int8(PyObject* self, PyObject* args) {
    PyArrayObject *input_array;
    PyObject *bit_indices_obj, *probabilities_obj;
    int strategy;
    if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &input_array, &PyList_Type, &bit_indices_obj, &PyList_Type, &probabilities_obj, &strategy)) {
        return NULL;
    }

    std::vector<int> bit_indices;
    std::vector<double> probabilities;

    for (Py_ssize_t i = 0; i < PyList_Size(bit_indices_obj); ++i) {
        PyObject* item = PyList_GetItem(bit_indices_obj, i);
        bit_indices.push_back(PyLong_AsLong(item));
    }

    for (Py_ssize_t i = 0; i < PyList_Size(probabilities_obj); ++i) {
        PyObject* item = PyList_GetItem(probabilities_obj, i);
        probabilities.push_back(PyFloat_AsDouble(item));
    }

    int num_elements = (int)PyArray_SIZE(input_array);
    int8_t* data = (int8_t*)PyArray_DATA(input_array);

    modify_bits(data, num_elements, bit_indices, probabilities, strategy);

    Py_INCREF(input_array);
    return (PyObject*)input_array;
}

// Wrapper function for Python (int8 Silent Modify)
static PyObject* silent_modify_bits_int8(PyObject* self, PyObject* args) {
    PyArrayObject *input_array;
    PyObject *bit_indices_obj, *probabilities_obj;
    double top_percent;
    int strategy;
    if (!PyArg_ParseTuple(args, "O!O!O!di", &PyArray_Type, &input_array, &PyList_Type, &bit_indices_obj, &PyList_Type, &probabilities_obj, &top_percent, &strategy)) {
        return NULL;
    }

    std::vector<int> bit_indices;
    std::vector<double> probabilities;

    for (Py_ssize_t i = 0; i < PyList_Size(bit_indices_obj); ++i) {
        PyObject* item = PyList_GetItem(bit_indices_obj, i);
        bit_indices.push_back(PyLong_AsLong(item));
    }

    for (Py_ssize_t i = 0; i < PyList_Size(probabilities_obj); ++i) {
        PyObject* item = PyList_GetItem(probabilities_obj, i);
        probabilities.push_back(PyFloat_AsDouble(item));
    }

    int num_elements = (int)PyArray_SIZE(input_array);
    int8_t* data = (int8_t*)PyArray_DATA(input_array);

    silent_modify_bits(data, num_elements, bit_indices, probabilities, strategy, top_percent);

    Py_INCREF(input_array);
    return (PyObject*)input_array;
}

static PyMethodDef methods[] = {
    {"modify_bits_int8", modify_bits_int8, METH_VARARGS, "Modify int8 bits based on flexible parameters."},
    {"silent_modify_bits_int8", silent_modify_bits_int8, METH_VARARGS, "Silent modify int8 bits, skip top X% based on absolute values."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "bitmod",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_bitmod(void) {
    import_array();
    return PyModule_Create(&module);
}
