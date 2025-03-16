// Struct to hold weight matrices
struct WeightMatrices {
    1: list<list<double>> V,
    2: list<list<double>> W
}

// Service definition for compute node
service compute {
    // Initialize and train an MLP model with given weights and training file
    WeightMatrices trainMLP(1: WeightMatrices weights, 2: string data 3: double eta, 4: i32 epochs),
    
    // Determine whether tasks are rejected
    bool rejectTask();
}