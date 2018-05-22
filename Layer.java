/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author pankaj
 */
import java.util.Random;
abstract class Layer
{
    protected Vec activation;
    protected int mInput_size;
    protected int mOutput_size;
    
    Layer(int inputs, int outputs)
    {
        mInput_size = inputs;
        mOutput_size = outputs;
        activation = new Vec(mOutput_size);   
    }
    
    abstract public void activate(Vec weights, Vec x);
}

class LayerLinear extends Layer
{
    LayerLinear(int inputs, int outputs)
    {
        super(inputs, outputs);
    }
    
    @Override
    public void activate(Vec weights, Vec x)
    {
        Vec M = new Vec(weights, 0, mInput_size*mOutput_size);
        Vec b = new Vec(weights, mInput_size*mOutput_size, mOutput_size);
        
        for(int r =0 ; r< mOutput_size; r++)
        {
            double value = 0;
            for(int c = 0; c < mInput_size; c++)
            {
                value += x.get(c)*M.get(r*mInput_size + c);    
            }
            activation.set(r, value);
        }
        activation.add(b);
    }

    private static Matrix vecCrossMultiply(Vec x, Vec y)
    {
        Matrix ret = new Matrix(0, y.size());

        for (int r= 0; r<x.size(); r++)
        {
            double [] products =new double[y.size()];
            for(int c=0; c<x.size(); c++)
            {
                products[c] = x.get(r)*y.get(c);
            }
            ret.takeRow(products);
        }
        return ret;
    }
    
    public static void vecCrossMultiplyAdd(Vec x, Vec y, Matrix addTo)
    {       
        if( addTo.cols() != y.size() || addTo.rows() != x.size())
        {
            System.err.println("Size matching error: addTo.cols() != y.size() || addTo.rows() != x.size()");
            return;
        }
        
        for(int r = 0; r < x.size(); r++)            
        {
            Vec row = addTo.row(r);
            for(int c= 0; c< y.size(); c++)
            {
                double value = x.get(r)*y.get(c);
                row.set(c, value + row.get(c));
                
            }
        }    
    }
    
    private static Vec getColumnwiseMean(Matrix m)
    {
        Vec v = new Vec(m.cols());
        for (int i =0; i< m.cols(); i++)
        {
            v.set(i,m.columnMean(i));            
        }
        return v;   
    }

    public void ordinary_least_squares(Matrix X, Matrix Y, Vec weights)
    {
        Matrix y_yMean_cross_x_xMean = new Matrix(mOutput_size, mInput_size);
        Matrix x_xMean_cross_x_xMean = new Matrix(mInput_size, mInput_size);
        
        Vec xMean = getColumnwiseMean(X);
        Vec yMean = getColumnwiseMean(Y);
        
        for (int i=0; i<X.rows(); i++)
        {
            Vec row_x = new Vec(0);
            row_x.copy(X.row(i));
            Vec row_y = new Vec(0);
            row_y.copy(Y.row(i));
            
            // Calculate Means first, then subtract from the original value
            for(int c = 0; c<row_x.size(); c++)
            {
                row_x.set(c, row_x.get(c)- xMean.get(c));
            }
            
            for(int c=0; c<row_y.size(); c++)
            {
                row_y.set(c, row_y.get(c));
            }
            
            vecCrossMultiplyAdd(row_y, row_x, y_yMean_cross_x_xMean);
            vecCrossMultiplyAdd(row_x, row_x, x_xMean_cross_x_xMean);
        }
        
        x_xMean_cross_x_xMean = x_xMean_cross_x_xMean.pseudoInverse();
        
        Matrix M = Matrix.multiply(y_yMean_cross_x_xMean, x_xMean_cross_x_xMean, false, false);
        //System.out.println("M-->\n"+M.toString());
        
        Vec B = new Vec(mOutput_size);
        
        for(int i = 0; i< mOutput_size; i++)
        {
            Vec row = M.row(i);
            B.set(i, yMean.get(i) - row.dotProduct(xMean));
        }
        //Storing M first
        for(int i = 0; i<M.rows(); i++)
        {
            for(int j = 0; j < M.cols(); j++)
            {       
                weights.set(i*M.cols()+j, M.row(i).get(j));
            }
        }
        //storing B after M in weights
        Vec bWeights = new Vec(weights, M.rows()*M.cols(), B.size());
        bWeights.add(B);
    }
    
/*    public static void OLSunittest_definitive() throws Exception
    {
        System.out.println("Perform in Linear Layer Ordinarry Least Square Unit Test....");
        System.out.println(".........Generate ramdom X and Weights........");
        Random rand = new Random();
        Vec weights = new Vec(3);
        
        weights.set(0, 3);
        weights.set(1, 1);
        weights.set(2,2);
        
        Matrix X = new Matrix(0,2);
        double [] row = X.newRow();
        row[0] = 1;
        row[1] = 8;
        row = X.newRow();
        row[0] = 5;
        row[1] = -1;
        row = X.newRow();
        row[0] = 1;
        row[1] = -2;
        
        Matrix Y = new Matrix(0,1);
        LayerLinear layer = new LayerLinear(2,1);
        double v = -0.5;
        
        for(int i =0; i < X.rows(); i++)
        {
            layer.activate(weights, X.row(i));
            row = Y.newRow();
            row[0] = layer.activation.get(0) + v++;
        }
        
        Vec weightsOLS = new Vec(3);
        layer.ordinary_least_squares(X, Y, weights);
        
        double SumSqrErr = weightsOLS.squaredDistance(weights);
        
        System.out.println("\nRandom X            =\n" + X.toString());
        System.out.println("\nRandom Weights(b, m)   = " +  weights);
        System.out.println("Weights Predicted      = " + weightsOLS.toString()+"\n");
        
        if(SumSqrErr > 20)
            throw new Exception("Sum Squared Error is greater than 20, error is " + SumSqrErr);
        else
            System.out.println("LinearLayer::ordinary_least_squares unit test, SSE = " + SumSqrErr);
        
        System.out.println("hello");    
    }
*/    
    
    public static void OLSUnitTest() throws Exception
    {
        System.out.println("Executing Ordinary Least Squares unit test in LinearLayer...");
        Random rand = new Random();
        Vec weights = new Vec(3);
        

        Matrix Y = new Matrix(0,1);
        LayerLinear layer = new LayerLinear(2,1);
        
        Matrix X = new Matrix(0,2);
        double[] row = X.newRow();
        row[0] = rand.nextInt(5);
        row[1] = rand.nextInt(5);
        row = X.newRow();
        row[0] = rand.nextInt(5);
        row[1] = rand.nextInt(5);
        row = X.newRow();
        row[0] = rand.nextInt(5);
        row[1] = rand.nextInt(5);
        
        weights.set(0, rand.nextInt(5));
        weights.set(1, rand.nextInt(5));
        weights.set(2, rand.nextInt(5));


        
        for(int i = 0; i< X.rows(); i++)
        {
            layer.activate(weights, X.row(i));
            row = Y.newRow();
            row[0] = layer.activation.get(0)+ rand.nextInt(2);
        }
        
        Vec weightsOLS = new Vec(3);
        layer.ordinary_least_squares(X, Y, weightsOLS);
        
        double SumSqrErr = weightsOLS.squaredDistance(weights);
        
        System.out.println("\nRandom X            =\n" +  X.toString());
        System.out.println("\nRandomly choosen Weights(b, m)   = " +  weights);
        System.out.println("Predicted Weights    = " + weightsOLS.toString()+"\n");

        if(SumSqrErr>15)
            throw new Exception("Error!! Thhe Sum Squared Error is greater than 15, error is " + SumSqrErr);
                else
            System.out.println("LinearLayer::Ordinary Least Squares unit test:: root-mean-squared-error = " + SumSqrErr);

        System.out.println("");
    }
    
/*    public static void ordaniary_least_square_unit_test(double standard_deviation)
    {
        Random rndm = new Random();
        System.out.println("OSL unit test...");
        System.out.println("1. Generate some random weights.");
        
        int test_row_x = 3, test_colm_x = 2, test_row_y = test_row_x, test_colm_y = 1; 
        Vec weights = new Vec(test_row_x);
        int valueBound = 25;
        for(int i = 0; i < weights.size(); i++)
        {
            weights.set(i, rndm.nextInt(valueBound));
        }
        
        System.out.println("2. Generate a random feature matrix, X.");
        Matrix X = new Matrix(0,test_colm_x);
        
        for(int i = 0; i < test_row_x; i++)
        {
            double [] row = X.newRow();
            row[0] = rndm.nextInt(valueBound);
            row[1] = rndm.nextInt(valueBound);
        }
        
        System.out.println("3. Use your LinearLayer.activate to compute a corresponding label matrix, Y.");
        Matrix Y_calculated_from_weight = new Matrix(0, test_colm_y);
        LayerLinear layer = new LayerLinear(test_colm_x, test_colm_y);
        
        for(int i = 0; i < X.rows(); i++)
        {
            layer.activate(weights, X.row(i));
            Vec copy_activation = new Vec(0);
            copy_activation.copy(layer.activation);            
        }
        System.out.println("\nRandom X            =\n" + X.toString());
        System.out.println("\nRandom Weights(b, m)   = " +  weights);
//        System.out.println("Weights Predicted      = " + weightsOLS.toString()+"\n");

    }
    
    
/*        public static void ordinary_least_squares_unit_test( double standard_deviation )
    {
        Random rndm = new Random();
        System.out.println("OSL unit test...");
        
        System.out.println("1. Generate some random weights.");
        
        int test_row_x = 3, test_colm_x = 2, test_row_y = test_row_x, test_colm_y = 1;
        Vec weights = new Vec(test_row_x); // Original Weights to compare
        int valueBound = 25;
        for(int i = 0; i < weights.size(); i++) {
            //Generates Random Number in the range 1 -20
            weights.set(i, rndm.nextInt(valueBound));
        }//end for loop

        System.out.println("2. Generate a random feature matrix, X.");
        Matrix X = new Matrix(0, test_colm_x);
        for(int r = 0; r < test_row_x; r++)
        {
            double[] row = X.newRow();
            row[0] = rndm.nextInt(valueBound);
            row[1] = rndm.nextInt(valueBound);
        }

        System.out.println("3. Use the LinearLayer.activate to compute a corresponding label matrix, Y.");
        Matrix Y_calculated_from_weight = new Matrix(0, test_colm_y);
        LayerLinear layer = new LayerLinear(test_colm_x, test_colm_y);
        for (int i = 0; i < X.rows(); i++) {
            layer.activate(weights, X.row(i));
            Vec copy_activation = new Vec(0);
            copy_activation.copy(layer.activation);
            Y_calculated_from_weight.takeRow(copy_activation.vals);
        }
        
        Matrix Y_noise = new Matrix(0, Y_calculated_from_weight.cols());
        double gauss_factor = (rndm).nextGaussian() * standard_deviation;
        System.out.println("4. Add a little random noise to Y. Gauss Fact:" + gauss_factor);
        for (int i = 0; i < X.rows(); i++) {
            double[] rowTemp = Y_noise.newRow();
            Vec row_y_weight = Y_calculated_from_weight.row(i);
            for(int c = 0 ; c < rowTemp.length; c++)
            {
                rowTemp[c] =  gauss_factor*row_y_weight.get(c) + row_y_weight.get(c);
            }
        }

        Vec calculated_weights_using_OLS = new Vec(test_row_x);
        System.out.println("6. Call your ordinary_least_squares method to generate new weights.");
        layer.ordinary_least_squares(X, Y_noise, calculated_weights_using_OLS);

        double SumSqrErr = calculated_weights_using_OLS.squaredDistance(weights);

        System.out.println("\nGenerate X=\n" + X.toString());
        System.out.println("\nCalculated Y=\n" + Y_calculated_from_weight.toString());
        System.out.println("\nCalculated Y with Noise=\n" + Y_noise.toString());
        System.out.println("\nGenerated Weights(M, B)= " +  weights.toString());
        System.out.println("\nPredicted Weights = " + calculated_weights_using_OLS.toString());

        if(SumSqrErr > 20)
            System.out.println("Sum Squared Error is greater than 20, SSE Value is:" + SumSqrErr);
        else
            System.out.println("Sum Squared Error(SSE) = " + SumSqrErr);

    }
    public static boolean unittest_activate()
    {
        Vec weights = new Vec(new double[]{1, 2, 3, 2, 1, 0, 1, 5});
        Vec x = new Vec(new double[]{0,1,2});

        LayerLinear ll = new LayerLinear(3,2);
        ll.activate(weights, x);

        Vec tru_activate = new Vec(2);
        tru_activate.set(0,9);
        tru_activate.set(1,6);

        return tru_activate.get(0) == ll.activation.get(0) && tru_activate.get(1) == ll.activation.get(1);
    }    
 */   
    
    }
    
    
        
        
        
                
 //   }        




 


