#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <ctime>

class ProgressBar {
private:
    std::size_t total;                  // Total item count
    std::size_t current;                // Current progress
    std::chrono::steady_clock::time_point start_time; // Start time
    std::chrono::steady_clock::time_point last_update_time; // Last display update time
    std::string prefix;                 // Prefix string (e.g. "Tree 1: ")
    bool force_display;                 // Force display on next display call
    double min_update_interval;         // Minimum display update interval (seconds)
    std::string start_time_str;         // Start time string
    int verbose_level;                  // Display level (0=silent, 1=normal, 2=verbose)
    int update_count;                   // Update count
    bool first_display_done;            // First display done flag
    
    // Convert time to format including seconds (display seconds to 3 decimal places)
    std::string format_time(double seconds) const {
        int total_seconds = int(seconds);
        int hours = total_seconds / 3600;
        int minutes = (total_seconds % 3600) / 60;
        double secs = seconds - hours * 3600 - minutes * 60;

        std::stringstream ss;
        if (hours > 0) {
            ss << hours << "h";
        }
        if (hours > 0 || minutes > 0) {
            ss << minutes << "m";
        }
        ss << std::fixed << std::setprecision(3) << secs << "s";

        return ss.str();
    }
    
    // Generate progress bar string
    std::string get_progress_bar() const {
        // Progress bar omitted for simplicity
        return "";
    }
    
public:
    double total_time;
    ProgressBar(std::size_t total, const std::string& prefix = "", int verbose_level = 1, double min_update_interval = 0.5)
        : total(total), current(0), prefix(prefix), force_display(true),
          min_update_interval(min_update_interval), verbose_level(verbose_level), update_count(0), first_display_done(false) {
        auto now = std::chrono::steady_clock::now();
        start_time = now;
        last_update_time = now;

        // Get current time in [HH:MM:SS] format
        auto now_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm now_tm = *std::localtime(&now_time_t);
        std::stringstream ss;
        ss << "[" << std::setfill('0') << std::setw(2) << now_tm.tm_hour
           << ":" << std::setfill('0') << std::setw(2) << now_tm.tm_min
           << ":" << std::setfill('0') << std::setw(2) << now_tm.tm_sec << "]";
        start_time_str = ss.str();
        
        // Display at start if verbose_level=2 (detailed mode)
        if (verbose_level >= 2) {
            display();
        }
    }
    
    ~ProgressBar() {
        // Add newline at end
        if (!force_display) {
            std::cerr << std::endl;
        }
    }
    
    // Update progress
    void update(std::size_t n = 1) {
        current += n;
        if (current > total) {
            current = total;
        }

        update_count++;

        // Adjust display frequency according to display level
        if (verbose_level >= 2) {
            // verbose_level=2: detailed mode - display all updates
            display();
        } else if (verbose_level == 1) {
            if (current >= total) {
                // Always display on completion
                display();
            } else if (update_count >= 3 && !first_display_done) {
                // 3 or more updates and not displayed yet
                auto now = std::chrono::steady_clock::now();
                double elapsed_since_start = std::chrono::duration<double>(now - start_time).count();
                
                // Display if 1 second or more has elapsed
                if (elapsed_since_start >= 1.0) {
                    display();
                }
            }
        }
        // verbose_level=0: silent mode - don't display
    }
    
    // Display current progress
    void display() {
        auto now = std::chrono::steady_clock::now();

        // Calculate elapsed time since last display
        double seconds_since_last_update = std::chrono::duration<double>(now - last_update_time).count();

        // Update display only if force display flag is true or minimum interval has elapsed
        if (force_display || seconds_since_last_update >= min_update_interval || current >= total) {
            double elapsed = std::chrono::duration<double>(now - start_time).count();

            // Progress rate
            double progress = current / double(total);
            double percent = progress * 100.0;

            // Estimate remaining time
            double eta = (progress > 0) ? (elapsed / progress - elapsed) : 0;

            // Convert current progress status to string
            std::stringstream ss;
            // Return cursor to beginning of line with "\r" and clear to end of line with "\033[K"
            ss << "\r\033[K" << start_time_str << " " << prefix;
            
            // Display "..." when total=1 and current=0
            if (total == 1 && current == 0) {
                ss << " ...";
            }
            // Don't display progress rate and count when total=1 and current=1
            else if (!(total == 1 && current == 1)) {
                // Display "DONE" when 100.0%
                if (progress >= 1.0) {
                    ss << " DONE ";
                } else {
                    ss << " " << std::fixed << std::setprecision(1) << percent << "% ";
                }
                ss << "(" << current << "/" << total << ")";
            }

            // Time display: show only elapsed time when progress is 100%, otherwise show elapsed and remaining time
            if (0.0 < progress) {
                ss << " " << format_time(elapsed);
                if (progress < 1.0) {
                    ss << "/" << format_time(eta);
                }
            }

            // Output to stderr and flush (display update immediately)
            std::cerr << ss.str() << std::flush;
            force_display = false;
            first_display_done = true;

            // Update last update time
            last_update_time = now;
        }
    }
    
    // Set progress to specific value
    void set_progress(std::size_t current_progress) {
        current = current_progress;
        if (current > total) {
            current = total;
        }
        display();
    }
    
    // Change prefix
    void set_prefix(const std::string& new_prefix) {
        prefix = new_prefix;
    }
    
    // Change total count
    void set_total(std::size_t new_total) {
        total = new_total;
    }
    
    // Check if completed
    bool is_completed() const {
        return current >= total;
    }
    
    // Process on completion
    void complete() {
        current = total;
        
        // Display only if not verbose_level=0 (silent mode)
        if (verbose_level > 0) {
            display();
            std::cerr << std::endl;  // Add newline
        }
        
        total_time = std::chrono::duration<double>(last_update_time - start_time).count();
        force_display = true;  // Reset for next display
    }
};

#endif // PROGRESS_BAR_HPP