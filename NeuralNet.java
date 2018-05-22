
import java.util.ArrayList;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author pankaj
 */
public class NeuralNet  extends SupervisedLearner{
    //All layers
    ArrayList<Layer> mLayers = new ArrayList<>();
    
    //Weight for Each layer
    ArrayList<Vec> mWeights = new ArrayList<>();

    @Override
    String name() {
        return "NeuralNet";
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    void train(Matrix features, Matrix labels) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        
        // For each call to train, we would add one layer to the arraylist of layers
        // and one weight object to the arraylist of weights
        LayerLinear layer = new LayerLinear(features.cols(), labels.cols());
        int layerWeightSize = features.cols()*labels.cols() /*M Size*/ + labels.cols() /*B Size*/;
        Vec weight = new Vec(layerWeightSize);
        
        //Calculate the weight based on the features and labels
        layer.ordinary_least_squares(features, labels, weight);
        
        //Finally add the layer and add the weights to the list
        mLayers.add(layer);
        mWeights.add(weight);
    }

    @Override
    Vec predict(Vec in) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        
        // For now we are only using one layer, hence we will only get the weights of that one layer
        int index_of_layer = 0;
        
        //The the corresponding layer and weight
        Layer layer = mLayers.get(index_of_layer);
        Vec weight = mWeights.get(index_of_layer);
        
        layer.activate(weight, in);
        
        return layer.activation;
    }
}
