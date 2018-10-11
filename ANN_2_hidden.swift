import TensorFlow

typealias TensorFloat = Tensor<Float>

struct NeuralNetwork {
    // 输入层 -----> 隐藏层1 -----> 隐藏层2 -----> 输出层
    // 3node   w1    5node   w2    4node   w3   1node
    
    static let nodeLayer2: Int32 = 5
    static let nodeLayer3: Int32 = 4
    
    var weights1: TensorFloat = 2 * TensorFloat(randomUniform: [3, nodeLayer2]) - 1
    var weights2: TensorFloat = 2 * TensorFloat(randomUniform: [nodeLayer2, nodeLayer3]) - 1
    var weights3: TensorFloat = 2 * TensorFloat(randomUniform: [nodeLayer3, 1]) - 1
    
    // 训练数据
    mutating func train(training_inputData: TensorFloat, training_outputData: TensorFloat, epoch: Int)
    {
        for _ in 0..<epoch {
            
            // 预测值
            let (output_l2, output_l3, output_l4) = predict(inputs: training_inputData)
            
            // 计算 error
            let error_l4 = training_outputData - output_l4
            let delta_l4 = error_l4 * sigmoid_derivative(input: output_l4)

            let error_l3 = delta_l4 • weights3.transposed()
            let delta_l3 = error_l3 * sigmoid_derivative(input: output_l3)

            let error_l2 = delta_l3 • weights2.transposed()
            let delta_l2 = error_l2 * sigmoid_derivative(input: output_l2)

            // 调整权重
            let adjusment_l4_l3 = output_l3.transposed() • delta_l4
            let adjusment_l3_l2 = output_l2.transposed() • delta_l3.transposed()
            let adjusment_l2_l1 = training_inputData.transposed() • delta_l2

            // 反向传播(Backpropagation): 将误差值再传回神经网络, 并调整权重.
            weights1 += adjusment_l2_l1
            weights2 += adjusment_l3_l2
            weights3 += adjusment_l4_l3
        }
    }
    
    // 预测数据
    func predict(inputs: TensorFloat) -> (TensorFloat, TensorFloat, TensorFloat) {
        let output_l2 = sigmoid(inputs • self.weights1)
        let output_l3 = sigmoid(output_l2 • self.weights2)
        let output_l4 = sigmoid(output_l3 • self.weights3)
        return (output_l2, output_l3, output_l4)
    }
}

// sigmoid 的导数
// 描述了 sigmoid 曲线的梯度, 也就是变化率
func sigmoid_derivative (input: TensorFloat) -> TensorFloat {
    return input * (1 - input)
}

func main()
{
    // 初始化
    var network = NeuralNetwork()
    // 随机初始化权重
    print("初始化权重:\n ", network.weights1, "\n", network.weights2, "\n", network.weights3)
    
    // 指定训练集
    // 三个输入 一个输出
    let training_inputData = TensorFloat([[0, 0, 1],
                                          [1, 1, 1],
                                          [1, 0, 1],
                                          [0, 1, 1]])
    let training_outputData = TensorFloat([[0, 1, 1, 0]]).transposed()
    
    // 开始训练
    network.train(training_inputData: training_inputData,
                  training_outputData: training_outputData, epoch: 10000)

    print("最终权重:\n ", network.weights1, "\n", network.weights2, "\n", network.weights3)

    // 预测结果
    let (_, _, output) = network.predict(inputs: TensorFloat([[1, 1, 1]]))
    print("预测数据[[1, 1, 1]] -> ?", output)
}

main()
