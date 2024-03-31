find ..  ! -path "*third_party/*"  ! -path "*exp" ! -path "*build/*" -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \;
