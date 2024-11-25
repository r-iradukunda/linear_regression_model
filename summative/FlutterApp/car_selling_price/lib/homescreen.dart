import 'package:flutter/material.dart';
import 'dart:convert'; // For JSON encoding and decoding
import 'package:http/http.dart' as http; // For making HTTP requests
import 'package:lottie/lottie.dart'; // For Lottie animations

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // Controllers for input fields
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _fuelController = TextEditingController();
  final TextEditingController _kmDrivenController = TextEditingController();
  final TextEditingController _vehicleAgeController = TextEditingController();

  String? _predictionResult; // To store the prediction result

  // Function to fetch the predicted price from FastAPI
  Future<void> _fetchPrediction() async {
    String name = _nameController.text;
    String fuel = _fuelController.text;
    String kmDriven = _kmDrivenController.text;
    String vehicleAge = _vehicleAgeController.text;

    // Input validation
    if (name.isEmpty ||
        fuel.isEmpty ||
        kmDriven.isEmpty ||
        vehicleAge.isEmpty) {
      _showDialog("Error", "Please fill in all the fields.");
      return;
    }

    try {
      // Prepare the API request
      const String apiUrl =
          "https://linear-regression-model-0jqg.onrender.com";
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: json.encode({
          "name": name,
          "fuel": fuel,
          "km_driven": int.parse(kmDriven),
          "vehicle_age": int.parse(vehicleAge),
        }),
      );

      // Check response status
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _predictionResult = "Predicted Car Price: \$${data['Car_price']}";
        });
      } else {
        _showDialog("Error", "Failed to fetch prediction. Try again.");
      }
    } catch (e) {
      _showDialog("Error", "An error occurred: $e");
    }
  }

  // Function to show dialog box with a message
  void _showDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("OK"),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 70,
        backgroundColor: const Color.fromARGB(249, 184, 218, 131),
        title: const Center(
          child: Text(
            "Car Price Predictor",
            style: TextStyle(
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 15.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 40),
              // Lottie Animation
              Center(
                child: Lottie.asset(
                  'assets/car.json', // Path to your Lottie file
                  height: 200,
                  // width: 200,
                  fit: BoxFit.cover,
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                'Car Information:',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 10),
              // Name Input
              TextField(
                controller: _nameController,
                decoration: const InputDecoration(
                  labelText: 'Car Name',
                  hintText: 'Enter car model (e.g., Maruti Swift Dzire VDI)',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 10),
              // Fuel Input
              TextField(
                controller: _fuelController,
                decoration: const InputDecoration(
                  labelText: 'Fuel Type',
                  hintText: 'Enter fuel type (e.g., Diesel, Petrol)',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 10),
              // KM Driven Input
              TextField(
                controller: _kmDrivenController,
                decoration: const InputDecoration(
                  labelText: 'Kilometers Driven',
                  hintText: 'Enter kilometers driven (e.g., 50000)',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.number,
              ),
              const SizedBox(height: 10),
              // Vehicle Age Input
              TextField(
                controller: _vehicleAgeController,
                decoration: const InputDecoration(
                  labelText: 'Vehicle Age',
                  hintText: 'Enter age of vehicle in years (e.g., 5)',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.number,
              ),
              const SizedBox(height: 20),
              // Predict Button
              Center(
                child: ElevatedButton(
                  onPressed: _fetchPrediction,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color.fromRGBO(0, 56, 4, 1),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 20, vertical: 10),
                  ),
                  child: const Text(
                    'PREDICT',
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              // Display Prediction Result
              if (_predictionResult != null)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(15),
                  decoration: BoxDecoration(
                    color: const Color.fromARGB(249, 184, 218, 131),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    _predictionResult!,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.black,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
