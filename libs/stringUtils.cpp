//
// Created by Jiří Winter on 02.11.2025.
//

#include "stringUtils.h"


std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}