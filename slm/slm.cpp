#include <fstream>
#include <iostream>
#include <set>
#include <string>

int main() {
   // std::string filename;
   // std::cout << "Enter the file name: \n";
   // std::cin >> filename;

   std::ifstream inputFile("shakespeare.txt");

   if (!inputFile.is_open()) {
      std::cerr << "Error: Cannot open file" << std::endl;
      return 1;
   }

   std::string content((std::istreambuf_iterator<char>(inputFile)),
		       std::istreambuf_iterator<char>());

   std::set<char> vocab(content.begin(), content.end());
   std::cout << "Vocab size: " << vocab.size() << "\n";

   inputFile.close();

   return 0;
}
