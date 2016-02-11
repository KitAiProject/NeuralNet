import java.util.Random;

/**
 * Created by tomohiro on 2016/02/02.
 */
public class BackPro {
    static final int INPUT_NUM = 15;
    static final int MID_NUM = 15;
    static final int OUTPUT_NUM = 3;

    double inputWeight[][];
    double outputWeight[][];

    double input[];
    double mid[];
    double output[];

    double alpha=0.01;


    public BackPro() {
        double data[][] = {
                {0,1,0,
                 0,1,0,
                 0,1,0,
                 0,1,0,
                 0,1,0,
                },
                {1,1,1,
                 0,0,1,
                 1,1,1,
                 1,0,0,
                 1,1,1},
                {1,1,1,
                 0,0,1,
                 1,1,1,
                 0,0,1,
                 1,1,1},
                {1,0,1,
                 1,0,1,
                 1,1,1,
                 0,0,1,
                 0,0,1},
        };

        double answer[][] = {
                {0,0,1},
                {0,1,0},
                {0,1,1},
                {1,0,0}
        };

        double testInput1[] ={
                1,1,0,
                0,1,0,
                0,1,0,
                0,1,0,
                0,1,0
        };

        double testInput2[] ={
                1,1,1,
                0,0,1,
                1,1,0,
                1,0,0,
                0,1,1
        };
        NeuralNetInitialize();
        while (true) {
            double e = 0.0;
            for (int i = 0; i < 4; i++) {
                computeAnswer(data[i]);
                backPropagation(answer[i], data[i]);
                e += Error(answer[i]);
            }

            System.out.println(e);
            if (e < 0.004) {
                break;
            }
        }


        //for (int i=0;i<4;i++){
        computeAnswer(testInput1);
        for(int i=0;i<3;i++) {
            System.out.println("output:" + output[i]);
        }

        computeAnswer(testInput2);
        for(int i=0;i<3;i++) {
            System.out.println("output:" + output[i]);
        }
        //}
    }
    public void NeuralNetInitialize(){
        input = new double[INPUT_NUM];
        mid = new double[MID_NUM];
        output = new double[OUTPUT_NUM];

        Random rnd = new Random();

        //íÜä‘ëw->ì¸óÕëwÇÃÉCÉÅÅ[ÉW
        inputWeight = new double[MID_NUM][INPUT_NUM];

        for(int i=0;i<MID_NUM;i++){
            for(int j=0;j<INPUT_NUM;j++){
                //0à»è„1ñ¢ñûÇÃêîÇÃóêêîÇ≈èâä˙âªÇ∑ÇÈ
                inputWeight[i][j] = (Math.random()-0.5)*0.01;
            }
        }

        outputWeight = new double[OUTPUT_NUM][MID_NUM];

        for(int i=0;i<OUTPUT_NUM;i++){
            for(int j=0;j<MID_NUM;j++){
                outputWeight[i][j]=rnd.nextDouble()*0.1-0.05;
            }
        }
    }

    public double sigmoid(double num){
        //expÇÕeÇÃ-numèÊÇ∆Ç¢Ç§à”ñ°
        return 1.0 / (1.0 + Math.exp(-num));
    }

    public void computeAnswer(double data[]){
        for(int i=0;i<INPUT_NUM;i++){
            input[i]=(double)data[i];
        }

        //íÜä‘ëwÇÃèoóÕÇÃåvéZ
        for(int i=0;i<MID_NUM;i++){
            mid[i]=0;
            for(int j=0;j<INPUT_NUM;j++) {
                mid[i] += inputWeight[i][j]*input[j];
            }
            mid[i]=sigmoid(mid[i]);
        }

        //èoóÕëwÇÃåvéZ
        for(int i=0;i<OUTPUT_NUM;i++){
            output[i]=0;
            for(int j=0;j<MID_NUM;j++) {
                output[i] += outputWeight[i][j]*mid[j];
            }
            output[i]=sigmoid(output[i]);
        }
    }

    public void backPropagation(double[] answer,double[] data){
        //èoóÕëwÇÃåÎç∑
        double outputDelta[]=new double[OUTPUT_NUM];

        for(int i=0;i<OUTPUT_NUM;i++){
            outputDelta[i]=(answer[i] - output[i])*output[i]*(1-output[i]);
        }

        //âBÇÍëwÇÃåÎç∑
        double midDelta[]=new double[MID_NUM];

        for(int i=0;i<MID_NUM;i++){
            midDelta[i] = 0.0;
        }
        for(int i=0;i<MID_NUM;i++) {
            for(int j=0;j<OUTPUT_NUM;j++) {
                midDelta[i] += outputDelta[j]*outputWeight[j][i]*mid[i]*(1-mid[i]);
            }
        }

        for(int i=0;i<MID_NUM;i++) {
            for(int j=0;j<OUTPUT_NUM;j++) {
                outputWeight[j][i] += alpha*outputDelta[j]*mid[i];
            }
        }

        for(int i=0;i<INPUT_NUM;i++) {
            for(int j=0;j<MID_NUM;j++) {
                inputWeight[j][i] += alpha*midDelta[j]*data[i];
            }
        }
    }

    public double Error(double teach[]){
        double e = 0.0;
        for(int i=0; i<OUTPUT_NUM; i++){
            e += Math.pow(teach[i]-output[i], 2.0);
        }
        e /=OUTPUT_NUM;
        return e;
    }
}