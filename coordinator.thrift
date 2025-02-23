struct WeightMatrices {
    1: list<list<double>> V,
    2: list<list<double>> W
}

service coordinator { 
    double train(1:string dir, 2:i32 rounds, 3:i32 epochs, 
    4:i32 h, 5:i32 k, 6:double eta) 

    void parse_compute_nodes(self):

    def work_scheduling(self):
}