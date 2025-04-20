// frontend/src/components/Navbar.tsx
import React from 'react';
import { NavLink } from 'react-router-dom';

const Navbar: React.FC = () => {
  // Using primary colors defined in tailwind config, fallback to indigo
  const activeClassName = "bg-primary hover:bg-primary-dark text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-150";
  const inactiveClassName = "text-gray-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-150";

  return (
    <nav className="bg-gray-800 shadow-lg sticky top-0 z-50"> {/* Make sticky */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <NavLink to="/" className="flex-shrink-0 text-white font-bold text-xl hover:text-gray-200 transition-colors duration-150">
              OULAD Recommender
            </NavLink>
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                <NavLink
                  to="/" // Link to the Demo page
                  // Use end prop for NavLink to only match exact root path
                  end
                  className={({ isActive }) => isActive ? activeClassName : inactiveClassName}
                >
                  Demo
                </NavLink>
                <NavLink
                  to="/about" // Link to the About page
                  className={({ isActive }) => isActive ? activeClassName : inactiveClassName}
                >
                  How it Works
                </NavLink>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;