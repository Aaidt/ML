#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

template <typename T> std::vector<T> encode(std::string s) {}

int main() {
   std::ifstream inputFile("shakespeare.txt");

   if (!inputFile.is_open()) {
      std::cerr << "Error: Cannot open file" << std::endl;
      return 1;
   }

   std::string content((std::istreambuf_iterator<char>(inputFile)),
		       std::istreambuf_iterator<char>());

   std::set<char> vocab(content.begin(), content.end());
   std::cout << "Vocab size: " << vocab.size() << "\n";

   std::map<int, char> char_to_idx;

   int idx = 0;
   for (const char &c : vocab) {
      char_to_idx[idx] = c;
      idx++;
   }

   for (const auto &[key, value] : char_to_idx) {
      std::cout << key << "->" << value << "\n";
   }

   inputFile.close();

   return 0;
}
