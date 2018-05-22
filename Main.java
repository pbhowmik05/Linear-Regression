// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

class Main
{
	static void test(SupervisedLearner learner, String challenge)
	{
		int num_of_repetation = 5;
                int num_of_fold = 10;
                // Load the training data
		String fn = "data\\" + challenge;
		Matrix Features = new Matrix();
		Features.loadARFF(fn + "_features.arff");
		Matrix Labels = new Matrix();
		Labels.loadARFF(fn + "_labels.arff");

		// Measure and report accuracy
		learner.nFoldCrossValidate(Features, Labels, num_of_repetation, num_of_fold);
		//System.out.println("Misclassifications by " + learner.name() + " at " + challenge + " = " + Integer.toString(misclassifications) + "/" + Integer.toString(testFeatures.rows()));
	}

	public static void testLearner(SupervisedLearner learner)
	{
		test(learner, "housing");
		//test(learner, "vow");
		//test(learner, "soy");
	}

	public static void main(String[] args)
	{
            LayerLinear layer = new LayerLinear(2,1);
            try
            {
		//testLearner(new BaselineLearner());
               // testLearner(new NeuralNet());
                
	
                //testLearner(new RandomForest(50));
                //layer.OLSunittest_definitive();
                //layer.ordinary_least_squares_unit_test(.1);
                //layer.ordaniary_least_square_unit_test(.1);  
                layer.OLSUnitTest();

                
            }
            catch(Exception e)
            {
            }
            testLearner(new NeuralNet());
                
                
                
	}   
        
        
}
