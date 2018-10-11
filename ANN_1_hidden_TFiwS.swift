

import TensorFlow

struct ANNParamaters: ParameterAggregate
{
    var w1 = Tensor<Float>(randomNormal: [3, 5])
    var w2 = Tensor<Float>(randomNormal: [5, 1])
}

func train_ANN_2_hidden_TFiwS(_ parameters: inout ANNParamaters, epochCount: Int32)
{
    let training_inputData = Tensor<Float>([[0, 0, 1],
                                          [1, 1, 1],
                                          [1, 0, 1],
                                          [0, 1, 1]])
    let training_outputData = Tensor<Float>([[0, 1, 1, 0]]).transposed()
    
    let learningRate: Float = 1
    
    
    for _ in 0..<epochCount {
        
        // Forward pass
        let (out_hidden_layer, out_out_layer) = predict_ANN_2_hidden_TFiwS(inputs: training_inputData, parameters: parameters)
        
        // Backward pass
        let delta_o_h = out_out_layer - training_outputData
        let dw2 = out_hidden_layer.transposed() • delta_o_h

        let delta_h_i = delta_o_h * parameters.w2.transposed() * out_hidden_layer * (1 - out_hidden_layer)
        let dw1 = training_inputData.transposed() • delta_h_i

        let gradients = ANNParamaters(w1: dw1, w2: dw2)
        parameters.update(withGradients: gradients) { (param, grad) in
            param -= grad * learningRate
        }
    }
}

func predict_ANN_2_hidden_TFiwS(inputs: Tensor<Float>, parameters: ANNParamaters) ->
    (out_hidden_layer: Tensor<Float>, out_out_layer: Tensor<Float>)
{
    let out_hidden_layer = sigmoid(inputs • parameters.w1)
    let out_out_layer = sigmoid(out_hidden_layer • parameters.w2)
    return (out_hidden_layer, out_out_layer)
}

func launch_ANN_1_hidden_TFiwS()
{
    var parameters = ANNParamaters()
    
    // 训练数据
    train_ANN_2_hidden_TFiwS(&parameters, epochCount: 60000)
    
    // 开始预测
    let (_, predictios) = predict_ANN_2_hidden_TFiwS(inputs: [[0, 0, 1]], parameters: parameters)
    
    print(predictios)
}

launch_ANN_1_hidden_TFiwS()
