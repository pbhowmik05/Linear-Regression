
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

abstract class SupervisedLearner 
{
    /// Return the name of this learner
    abstract String name();

    /// Train this supervised learner
    abstract void train(Matrix features, Matrix labels);

    /// Make a prediction
    abstract Vec predict(Vec in);

    /// Measures the misclassifications with the provided test data
    int countMisclassifications(Matrix features, Matrix labels)
    {
        if(features.rows() != labels.rows())
            throw new IllegalArgumentException("Mismatching number of rows");
	int mis = 0;
        for(int i = 0; i < features.rows(); i++)
	{
            Vec feat = features.row(i);
            Vec pred = predict(feat);
            Vec lab = labels.row(i);
            for(int j = 0; j < lab.size(); j++)
            {
		if(pred.get(j) != lab.get(j))
		mis++;
            }
	}
            return mis;
    }
        
    double sumSquaredError(Matrix features, Matrix labels)
    {
        if(features.rows() != labels.rows())
            throw new IllegalArgumentException("Number of data doesn't match for features and labels");
            
        double sum_sq_err = 0;
        for(int i = 0; i < features.rows(); i++)
        {
            Vec feat = features.row(i);
            Vec pred = predict(feat);
            Vec lab = labels.row(i);
            for(int j = 0; j < lab.size(); j++)
            {
                sum_sq_err += Math.pow(pred.get(j)- lab.get(j), 2);
            }
        }
            return sum_sq_err;
    }
        
        // Implementing Fisher–Yates shuffle
        static void shuffleArray(int[] ar) {
        // If running on Java 6 or older, use `new Random()` on RHS here
        Random rnd = new Random();
            for (int i = ar.length - 1; i > 0; i--) {
                int index = rnd.nextInt(i + 1);
                // Simple swap
                int a = ar[index];
                ar[index] = ar[i];
                ar[i] = a;
            }
        }
        void nFoldCrossValidate(Matrix features, Matrix labels, int Repetition, int nFoldValue)
        {
            for (int i = 0; i < Repetition; i++) {
            double Root_Mean_Sqr_Error = 0;
			
            for (int k = 0; k < nFoldValue; k++) {
				
		Matrix features_to_train = new Matrix(0, features.cols());
		Matrix features_to_test = new Matrix(0, features.cols());
		Matrix labels_to_train = new Matrix(0, labels.cols());
		Matrix labels_to_test = new Matrix(0, labels.cols());
                
		ArrayList<Integer> Array_index = new ArrayList<Integer>();
		Random rand = new Random();		
		int no_of_test_set = features.rows() / nFoldValue;
		int no_of_data_set = features.rows();

		for (int j = 0; Array_index.size() < no_of_data_set; j++) 
                {
                    int Index = rand.nextInt(no_of_data_set);
                    if (Array_index.contains(Index) == false) 
                    {
			Array_index.add(Index);
                    }
		}
				
		int StartIndexValue = (i * no_of_test_set) % no_of_data_set;
		int EndIndexValue = (StartIndexValue + no_of_test_set) % no_of_data_set;

		for (int j = 0; j < no_of_data_set; j++) 
                {
                    Vec rowFeature = features.row(Array_index.get(j));
                    Vec rowLabel = labels.row(Array_index.get(j));
                    if (j >= StartIndexValue && j <= EndIndexValue) 
                    {
			features_to_test.takeRow(rowFeature.vals);
			labels_to_test.takeRow(rowLabel.vals);
                    } 
                    else 
                    {
			features_to_train.takeRow(rowFeature.vals);
			labels_to_train.takeRow(rowLabel.vals);
                    }
		}

		train(features_to_train, labels_to_train);
		Root_Mean_Sqr_Error += (sumSquaredError(features_to_test, labels_to_test)/no_of_test_set);
            }
			
            Root_Mean_Sqr_Error = Math.sqrt(Root_Mean_Sqr_Error/nFoldValue);
            
            System.out.println("The Root mean square error is for Running the fold " + i + " is " + Root_Mean_Sqr_Error);
	}
            //System.out.println("The Root mean square error is for Running the fold " + i + " is " + Root_Mean_Sqr_Error);

    }
}
